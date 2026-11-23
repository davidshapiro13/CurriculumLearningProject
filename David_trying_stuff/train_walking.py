"""Train the original 17-motor humanoid with PPO; no torso support.

Examples (from the project root):
    python LearningMujoco/train_walking.py --timesteps 1000000
    python LearningMujoco/train_walking.py --evaluate --video walking.mp4

An episode terminates on non-foot ground contact, or truncates at 1000
policy steps. Each policy step contains up to five 0.003-second physics steps.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


class WalkingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 67}

    def __init__(self, render_mode=None, model_path=None):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(model_path or HERE / "my_humanoid.xml"))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 5
        self.max_episode_steps = 1000
        self.floor_id = self.model.geom("floor").id
        self.foot_body_ids = {self.model.body(f"{side}_foot").id for side in ("left", "right")}
        self.joint_addresses = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        self.lower, self.upper = self.model.actuator_ctrlrange.T.copy()
        self.neutral = np.zeros(self.model.nu)
        if self.model.nu != 17 or np.any(self.model.actuator_biasprm):
            raise ValueError("WalkingEnv requires the original 17 torque motors")
        # Native motor commands, not angles. Gearing converts these into torque.
        self.action_space = spaces.Box(self.lower.astype(np.float32), self.upper.astype(np.float32))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.model.nq - 2 + self.model.nv + 22 * (self.model.nbody - 1) + self.model.nv - 6,), np.float32
        )
        self.renderer = None
        self.elapsed_steps = 0
        self._done = True

    def _observation(self):
        # Exclude absolute horizontal position; retain height, orientation, angles,
        # and all velocities. The free root remains subject to gravity.
        return np.concatenate((self.data.qpos[2:], self.data.qvel,
            self.data.cinert[1:].ravel(), self.data.cvel[1:].ravel(),
            self.data.qfrc_actuator[6:], self.data.cfrc_ext[1:].ravel())).astype(np.float32)

    def _center_of_mass(self):
        return np.average(self.data.xipos, axis=0, weights=self.model.body_mass)

    def fallen_body(self):
        """Return the first non-foot body touching the floor, in either pair order."""
        for contact in self.data.contact[:self.data.ncon]:
            # MuJoCo may report proximity contacts inside a positive margin.
            if contact.dist > 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 == self.floor_id:
                other = g2
            elif g2 == self.floor_id:
                other = g1
            else:
                continue
            body_id = int(self.model.geom_bodyid[other])
            if body_id not in self.foot_body_ids:
                return self.model.body(body_id).name
        return None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] += self.np_random.uniform(-0.01, 0.01, self.model.nq)
        self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])
        self.data.qvel[:] = self.np_random.uniform(-0.01, 0.01, self.model.nv)
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.start_x = float(self._center_of_mass()[0])
        self.elapsed_steps = 0
        self._done = False
        return self._observation(), {"elapsed_steps": 0}


    def step(self, action):

        def forward_alignment(body_name):
            # Body orientation: columns are its local axes in world coordinates.
            rotation = self.data.body(body_name).xmat.reshape(3, 3)

            # Project the body's local +X direction onto the horizontal plane.
            forward_xy = rotation[:2, 0]
            length = np.linalg.norm(forward_xy)

            if length < 1e-6:
                return 0.0

            # Alignment with world +X: +1 forward, 0 sideways, -1 backward.
            return float(forward_xy[0] / length)
        
        if self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Expected finite action with shape {self.action_space.shape}")
        action = np.clip(action, self.lower, self.upper)
        self.data.ctrl[:] = action
        x_before = float(self._center_of_mass()[0])
        time_before = float(self.data.time)
        fallen = self.fallen_body()
        for _ in range(self.frame_skip):
            if fallen is not None:
                break
            old_time = self.data.time
            mujoco.mj_step(self.model, self.data)
            if self.data.time <= old_time or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("MuJoCo became unstable; inspect controller gains and model")
            # Refresh contacts at the newly integrated pose, including short-lived
            # impacts between policy steps. Stop immediately on a forbidden contact.
            mujoco.mj_forward(self.model, self.data)
            fallen = self.fallen_body()
        self.elapsed_steps += 1
        dt = self.data.time - time_before
        forward_velocity = (float(self._center_of_mass()[0]) - x_before) / dt if dt > 0 else 0.0
        terminated = fallen is not None
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        # Standard Humanoid-style reward, while retaining the requested contact
        # termination instead of Gymnasium's torso-height termination.
        height = float(self.data.qpos[2])
        height_reward = 5.0 * np.exp(-((height - 1.4) / 0.15) ** 2)
        upright_reward = 0.0 if terminated else height_reward
        effort_cost = 0.2 * float(np.sum(action ** 2))

        pelvis_alignment = forward_alignment("pelvis")
        torso_alignment = forward_alignment("torso")
        left_foot_alignment = forward_alignment("left_foot")
        right_foot_alignment = forward_alignment("right_foot")

        heading_reward = (
            0.5 * pelvis_alignment
            + 0.5 * torso_alignment
            + 0.25 * left_foot_alignment
            + 0.25 * right_foot_alignment
)

        mujoco.mj_rnePostConstraint(self.model, self.data)
        contact_cost = min(5e-7 * float(np.sum(self.data.cfrc_ext ** 2)), 10.0)
        reward = 1.25 * forward_velocity + upright_reward + heading_reward - effort_cost - contact_cost
        self._done = terminated or truncated
        info = {
            "elapsed_steps": self.elapsed_steps,
            "forward_velocity": forward_velocity,
            "effort_cost": effort_cost,
            "upright_reward": upright_reward,
            "forward_distance": float(self._center_of_mass()[0]) - self.start_x,
            "fallen_body": fallen,
            "end_reason": "fall" if terminated else "time_limit" if truncated else None,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera="track")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def evaluate(policy, normalizer, model_path, episodes=5, video=None, seed=10000):
    """Evaluate fixed held-out seeds with frozen normalization and no exploration."""
    import imageio.v2 as imageio
    env = WalkingEnv(render_mode="rgb_array" if video else None, model_path=model_path)
    records = []
    writer = imageio.get_writer(str(video), fps=env.metadata['render_fps'], codec='libx264') if video else None
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            for _ in range(env.max_episode_steps):
                action, _ = policy.predict(normalizer.normalize_obs(obs), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if writer:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            records.append(dict(reward=total_reward, length=info['elapsed_steps'],
                                distance=info['forward_distance'], survived=int(truncated),
                                fallen_body=info['fallen_body']))
    finally:
        if writer:
            writer.close()
        env.close()
    return {key: float(np.mean([r[key] for r in records]))
            for key in ('reward', 'length', 'distance', 'survived')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                        help='Additional policy steps; PPO finishes complete rollouts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=HERE / 'walking_original_run')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', choices=('best', 'policy'), default='best')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--eval-every', type=int, default=50_000)
    parser.add_argument('--video', type=Path, default=None,
                        help='Optional evaluation video; omitted for headless metrics')
    args = parser.parse_args()
    if min(args.timesteps, args.episodes, args.eval_every) <= 0:
        parser.error('Step counts and episode counts must be positive')
    if args.resume and args.evaluate:
        parser.error('Choose either --resume or --evaluate')
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import torch
    torch.set_num_threads(1)

    model_path = args.output / 'model.xml'
    config_path = args.output / 'config.json'
    if args.evaluate or args.resume:
        if not config_path.exists() or json.loads(config_path.read_text()).get('version') != 2:
            parser.error('This is not an original-model run. Old 20-action checkpoints cannot be reused; start a new run.')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(HERE / 'my_humanoid.xml', model_path)
        config_path.write_text(json.dumps(dict(version=2, seed=args.seed, control='17 stock torque motors',
            max_episode_steps=1000, timestep=0.003, frame_skip=5,
            hyperparameters_source='https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L474',
            requested_timesteps=args.timesteps), indent=2))

    if args.evaluate:
        prefix = args.checkpoint
        normalizer = VecNormalize.load(str(args.output / f'{prefix}_normalization.pkl'),
                                       DummyVecEnv([lambda: WalkingEnv(model_path=model_path)]))
        normalizer.training = False
        normalizer.norm_reward = False
        try:
            policy = PPO.load(args.output / f'{prefix}.zip', device='cpu')
            stats = evaluate(policy, normalizer, model_path, args.episodes, args.video)
            print(json.dumps(stats, indent=2))
            if args.video:
                print(f'Saved {args.video}')
        finally:
            normalizer.close()
        return

    monitor_name = f'episodes_{time.time_ns()}'
    base = DummyVecEnv([lambda: Monitor(WalkingEnv(model_path=model_path),
                                       str(args.output / monitor_name))])
    if args.resume:
        env = VecNormalize.load(str(args.output / 'policy_normalization.pkl'), base)
        env.training = True
        policy = PPO.load(args.output / 'policy.zip', env=env, device='cpu')
    else:
        env = VecNormalize(base, gamma=0.95)
        # RL Zoo's tuned Humanoid-v4 PPO settings. Our contact termination differs,
        # so these are a grounded starting point, not a promise of solved walking.
        policy = PPO('MlpPolicy', env, seed=args.seed, device='cpu', verbose=0,
                     n_steps=512, batch_size=256, n_epochs=5,
                     learning_rate=3.56987e-5, gamma=0.95, gae_lambda=0.9,
                     ent_coef=0.00238306, clip_range=0.3, max_grad_norm=2,
                     vf_coef=0.431892,
                     policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                        activation_fn=torch.nn.ReLU, log_std_init=-2,
                                        ortho_init=False))

    class ProgressAndEvaluation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.start_steps = policy.num_timesteps
            self.started = time.monotonic()
            self.last_print = self.started
            self.next_eval = self.start_steps + args.eval_every
            score_path = args.output / 'best_metrics.json'
            self.best = json.loads(score_path.read_text())['reward'] if score_path.exists() else -np.inf

        def save_and_evaluate(self):
            stats = evaluate(self.model, env, model_path, args.episodes)
            stats['timesteps'] = self.model.num_timesteps
            path = args.output / 'evaluations.csv'
            exists = path.exists()
            with path.open('a', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=list(stats))
                if not exists:
                    writer.writeheader()
                writer.writerow(stats)
            print(f"Evaluation: {stats}", flush=True)
            self.model.save(args.output / 'policy')
            env.save(str(args.output / 'policy_normalization.pkl'))
            if stats['reward'] > self.best:
                self.best = stats['reward']
                self.model.save(args.output / 'best')
                env.save(str(args.output / 'best_normalization.pkl'))
                (args.output / 'best_metrics.json').write_text(json.dumps(stats, indent=2))
                print('Saved best policy with matching normalization.', flush=True)

        def _on_step(self):
            now = time.monotonic()
            if now - self.last_print >= 15:
                done = self.num_timesteps - self.start_steps
                rate = done / (now - self.started)
                remaining = max(0, args.timesteps - done) / max(rate, 1)
                lengths = [item['l'] for item in self.model.ep_info_buffer]
                print(f'{done:,}/{args.timesteps:,} additional steps | {rate:.0f} steps/s | '
                      f'ETA {remaining / 60:.1f} min | recent episode length '
                      f'{np.mean(lengths) if lengths else 0:.1f}', flush=True)
                self.last_print = now
            if self.num_timesteps >= self.next_eval:
                self.save_and_evaluate()
                self.next_eval = self.num_timesteps + args.eval_every
            return True

    callback = ProgressAndEvaluation()
    print(f'Training original humanoid. Run directory: {args.output}', flush=True)
    try:
        policy.learn(total_timesteps=args.timesteps, callback=callback,
                     reset_num_timesteps=not args.resume)
        callback.save_and_evaluate()
    except KeyboardInterrupt:
        print('Interrupted; saving current policy and normalization.', flush=True)
    finally:
        policy.save(args.output / 'policy')
        env.save(str(args.output / 'policy_normalization.pkl'))
        env.close()
    print(f'Saved training artifacts to {args.output}')


if __name__ == '__main__':
    main()
