"""Physics and episode rules. Reward design lives in rewards.py."""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from .rewards import calculate_reward


def stock_model_path():
    return Path(gym.__file__).parent / 'envs/mujoco/assets/humanoid.xml'


class RunningEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 67}

    def __init__(self, config, model_path=None, render_mode=None, reward_function=calculate_reward):
        super().__init__()
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Only rgb_array rendering is supported')
        self.config = config
        self.task = config['environment']
        self.reward_function = reward_function
        self.model = mujoco.MjModel.from_xml_path(str(model_path or stock_model_path()))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.renderer = None
        self.metadata = dict(self.metadata, render_fps=round(1 / (self.task['frame_skip'] * self.model.opt.timestep)))
        self.floor = self.model.geom('floor').id
        self.feet = {self.model.body(f'{s}_foot').id for s in ('left', 'right')}
        self.native_controls = self.task.get('control_mode') == 'native'
        self.extended = self.task.get('observation_mode', 'extended') == 'extended'
        # Rhythm/smoothness objectives need their corresponding observable state.
        if config['rewards']['knee_weight'] or config['rewards']['action_change_cost']:
            self.extended = True
        self.action_space = (spaces.Box(self.model.actuator_ctrlrange[:, 0].astype(np.float32),
                                        self.model.actuator_ctrlrange[:, 1].astype(np.float32))
                             if self.native_controls else spaces.Box(-1, 1, (self.model.nu,), dtype=np.float32))
        base_size = self.model.nq - 2 + self.model.nv + 22 * (self.model.nbody - 1) + self.model.nv - 6
        # Add distance/lateral offset, gait phase, and previous action for Markov state.
        self.observation_space = spaces.Box(-np.inf, np.inf, (base_size + (4 + self.model.nu if self.extended else 0),), dtype=np.float32)
        self.previous_action = np.zeros(self.model.nu)
        self.start = np.zeros(3)
        self.phase_offset = 0.0
        self.steps = 0
        self.done = True

    def center(self):
        return np.average(self.data.xipos, axis=0, weights=self.model.body_mass)

    def foot_offsets(self):
        pelvis_x = float(self.data.body('pelvis').xpos[0])
        return [float(self.data.geom(f'{side}_foot').xpos[0]) - pelvis_x
                for side in ('left', 'right')]

    def swing_feet(self, before_offsets, dt):
        contacts = set()
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist <= 0:
                a, b = int(contact.geom1), int(contact.geom2)
                other = b if a == self.floor else a if b == self.floor else None
                if other is not None:
                    contacts.add(int(self.model.geom_bodyid[other]))
        feet = []
        for index, side in enumerate(('left', 'right')):
            geom_id = self.model.geom(f'{side}_foot').id
            ahead = float(self.data.geom_xpos[geom_id, 0] - self.data.body('pelvis').xpos[0])
            # Stock humanoid feet are spheres; subtract radius for bottom height.
            clearance = float(self.data.geom_xpos[geom_id, 2] - self.model.geom_size[geom_id, 0]
                              - self.data.geom_xpos[self.floor, 2])
            feet.append(dict(ahead=ahead, clearance=clearance,
                             contact=self.model.body(f'{side}_foot').id in contacts,
                             relative_vx=(ahead - before_offsets[index]) / dt if dt > 0 else 0.0))
        return feet

    def phase(self):
        return self.phase_offset + 2 * np.pi * self.task['gait_frequency_hz'] * self.data.time

    def observation(self):
        position = self.center() - self.start
        base = np.concatenate((self.data.qpos[2:], self.data.qvel,
            self.data.cinert[1:].ravel(), self.data.cvel[1:].ravel(),
            self.data.qfrc_actuator[6:], self.data.cfrc_ext[1:].ravel()))
        if not self.extended:
            return base.astype(np.float32)
        return np.concatenate((base,
            [position[0] / self.task['finish_distance'], position[1], np.sin(self.phase()), np.cos(self.phase())],
            self.previous_action)).astype(np.float32)

    def fallen_body(self):
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist > 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a == self.floor else a if b == self.floor else None
            if other is not None:
                body = int(self.model.geom_bodyid[other])
                if body not in self.feet:
                    return self.model.body(body).name
        return None

    def outcome(self):
        fallen = self.fallen_body()
        # Falls take precedence over a simultaneous finish-line crossing.
        success = fallen is None and self.center()[0] - self.start[0] >= self.task['finish_distance']
        return fallen, bool(success)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.task.get('reset_style') in ('state_noise', 'original'):
            noise = self.task['reset_noise']
            self.data.qpos[:] += self.np_random.uniform(-noise, noise, self.model.nq)
            self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])
            self.data.qvel[:] = self.np_random.uniform(-noise, noise, self.model.nv)
        else:
            joints = self.model.actuator_trnid[:, 0]
            addresses = self.model.jnt_qposadr[joints]
            noise = self.np_random.uniform(-self.task['reset_noise'], self.task['reset_noise'], self.model.nu)
            self.data.qpos[addresses] = np.clip(self.data.qpos[addresses] + noise, *self.model.jnt_range[joints].T)
        # Keep the root upright at its original start; randomize which leg leads.
        self.phase_offset = float(self.np_random.uniform(0, 2 * np.pi))
        self.previous_action[:] = 0
        self.steps = 0
        self.done = False
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        self.start = self.center().copy()
        return self.observation(), {'distance': 0.0}

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset the environment before stepping')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError('Expected 17 finite normalized motor commands')
        action = np.clip(action, self.action_space.low, self.action_space.high)
        low, high = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = action if self.native_controls else low + (action + 1) * 0.5 * (high - low)
        before, start_time = self.center().copy(), float(self.data.time)
        before_feet = self.foot_offsets()
        fallen, success = self.outcome()
        for _ in range(self.task['frame_skip']):
            if fallen is not None or success:
                break
            old_time = float(self.data.time)
            mujoco.mj_step(self.model, self.data)
            if self.data.time <= old_time or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError('Unstable MuJoCo simulation')
            mujoco.mj_forward(self.model, self.data)
            fallen, success = self.outcome()
        mujoco.mj_rnePostConstraint(self.model, self.data)
        self.steps += 1
        after = self.center()
        dt = float(self.data.time) - start_time
        velocity = (after - before) / dt if dt > 0 else np.zeros(3)
        torso = self.data.body('torso').xmat.reshape(3, 3)
        # Unnormalized horizontal projection also discourages pointing vertically.
        heading = np.mean([self.data.body(name).xmat.reshape(3, 3)[0, 0] for name in ('torso', 'pelvis')])
        state = dict(dt=dt, dx=float(after[0] - before[0]), vx=float(velocity[0]), vy=float(velocity[1]),
                     height=float(self.data.qpos[2]), torso_up=float(torso[2, 2]), heading=float(heading),
                     lateral_distance=float(after[1] - self.start[1]), phase=self.phase(),
                     knees=[float(self.data.joint(f'{side}_knee').qpos[0]) for side in ('left', 'right')],
                     action=action, previous_action=self.previous_action.copy(), fell=fallen is not None, success=success)
        body_headings = {}
        for name in ('torso', 'pelvis', 'left_foot', 'right_foot'):
            xy = self.data.body(name).xmat.reshape(3, 3)[:2, 0]
            length = np.linalg.norm(xy)
            body_headings[name] = float(xy[0] / length) if length >= 1e-6 else 0.0
        state.update(body_headings=body_headings, motor_controls=self.data.ctrl.copy(),
                     swing_feet=self.swing_feet(before_feet, dt),
                     contact_force_squared=float(np.sum(self.data.cfrc_ext ** 2)))
        reward, terms = self.reward_function(state, self.config['rewards'])
        terminated = fallen is not None or success
        truncated = self.steps >= self.task['max_episode_steps'] and not terminated
        self.done = terminated or truncated
        self.previous_action[:] = action
        info = dict(distance=float(after[0] - self.start[0]), steps=self.steps, success=success,
                    fallen_body=fallen, end_reason='fall' if fallen is not None else 'finish' if success else 'timeout' if truncated else None,
                    reward_terms=terms)
        return self.observation(), reward, bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        self.renderer.update_scene(self.data, camera='track')
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
