"""Run: python -m unittest Training_run.test_running"""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import mujoco
import numpy as np
from .environment import RunningEnv
from .evaluation import episode_score, policy_score, saved_reward
from .rewards import calculate_reward
from .settings import load_config
from .train import train


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).with_name('legacy_config.json'))
        self.env = RunningEnv(self.config)
        self.env.reset(seed=1)
        self.addCleanup(self.env.close)

    def test_stock_model_and_reset(self):
        self.assertEqual(self.env.model.nu, 17)
        self.assertEqual(self.env.model.neq, 0)
        obs, _ = self.env.reset(seed=4)
        self.assertEqual(obs.shape, (369,))
        self.env.step(np.zeros(17))
        repeated, _ = self.env.reset(seed=4)
        np.testing.assert_array_equal(obs, repeated)
        np.testing.assert_array_equal(self.env.data.qpos[:3], self.env.model.qpos0[:3])
        self.assertIsNone(self.env.fallen_body())

    def test_contact_definition(self):
        floor = self.env.floor
        foot = self.env.model.geom('right_foot').id
        hand = self.env.model.geom('right_hand').id
        for pair, distance, fails in [((floor, foot), 0, False), ((foot, floor), -0.01, False),
                                       ((hand, floor), 0, True), ((floor, hand), -0.01, True),
                                       ((floor, hand), 0.001, False), ((foot, hand), -0.01, False)]:
            contact = SimpleNamespace(geom1=pair[0], geom2=pair[1], dist=distance)
            with patch.object(self.env, 'data', SimpleNamespace(ncon=1, contact=[contact])):
                self.assertEqual(self.env.fallen_body() is not None, fails)

    def test_finish_is_forward_displacement_not_sideways_or_path_length(self):
        self.env.data.qpos[1] += 21
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertFalse(self.env.outcome()[1])
        self.env.data.qpos[0] += 20.01
        mujoco.mj_forward(self.env.model, self.env.data)
        _, _, terminated, truncated, info = self.env.step(np.zeros(17))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['success'])
        self.assertEqual(info['reward_terms']['finish'], 500)
        with self.assertRaises(RuntimeError):
            self.env.step(np.zeros(17))

    def test_fall_overrides_simultaneous_finish(self):
        self.env.start[0] -= 21
        with patch.object(self.env, 'fallen_body', return_value='torso'):
            _, _, terminated, truncated, info = self.env.step(np.zeros(17))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['end_reason'], 'fall')
        self.assertEqual(info['reward_terms']['fall'], -200)
        self.assertEqual(info['reward_terms']['finish'], 0)

    def test_short_contact_during_action(self):
        with patch.object(self.env, 'fallen_body', side_effect=[None, None, 'left_shin']):
            _, _, terminated, _, info = self.env.step(np.zeros(17))
        self.assertTrue(terminated)
        self.assertEqual(info['fallen_body'], 'left_shin')
        self.assertAlmostEqual(self.env.data.time, 2 * self.env.model.opt.timestep)

    def test_timeout_and_boundary_precedence(self):
        self.env.task['max_episode_steps'] = 2
        self.assertFalse(self.env.step(np.zeros(17))[3])
        self.assertTrue(self.env.step(np.zeros(17))[3])
        self.env.reset(seed=1)
        self.env.steps = 1
        with patch.object(self.env, 'fallen_body', return_value='torso'):
            _, _, terminated, truncated, _ = self.env.step(np.zeros(17))
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_control_mapping(self):
        for value, expected in [(-1, -0.4), (0, 0), (1, 0.4)]:
            self.env.reset(seed=1)
            self.env.step(np.full(17, value))
            np.testing.assert_allclose(self.env.data.ctrl, expected)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.w = load_config(Path(__file__).with_name('legacy_config.json'))['rewards']
        self.state = dict(dt=.015, dx=.045, vx=3., vy=0., height=1.4, torso_up=1., heading=1.,
                          lateral_distance=0., phase=np.pi/2, knees=np.deg2rad([-10, -65]),
                          action=np.zeros(17), previous_action=np.zeros(17), fell=False, success=False)

    def test_both_knees_and_alternation(self):
        _, perfect = calculate_reward(self.state, self.w)
        _, dragging = calculate_reward(dict(self.state, knees=np.deg2rad([-10, -10])), self.w)
        _, alternate = calculate_reward(dict(self.state, phase=3*np.pi/2, knees=np.deg2rad([-65, -10])), self.w)
        _, stationary = calculate_reward(dict(self.state, vx=0), self.w)
        self.assertGreater(perfect['knees'], dragging['knees'])
        self.assertAlmostEqual(perfect['knees'], alternate['knees'])
        self.assertEqual(stationary['knees'], 0)

    def test_heading_sideways_and_terminal_rewards(self):
        good, _ = calculate_reward(self.state, self.w)
        bad, _ = calculate_reward(dict(self.state, heading=-1, vy=2, lateral_distance=2), self.w)
        self.assertGreater(good, bad)
        self.assertAlmostEqual(calculate_reward(dict(self.state, fell=True), self.w)[0], good - 200)
        self.assertAlmostEqual(calculate_reward(dict(self.state, success=True), self.w)[0], good + 500)

    def test_ranking_prioritizes_finishing(self):
        self.assertGreater(policy_score(dict(success=1, distance=20, reward=1)),
                           policy_score(dict(success=0, distance=19, reward=1000)))
        self.assertGreater(episode_score(dict(success=True, steps=100, reward=0)),
                           episode_score(dict(success=True, steps=200, reward=999)))


class TrainingTests(unittest.TestCase):
    def test_episode_budget_updates_and_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config['environment']['max_episode_steps'] = 20
            config['ppo'].update(n_steps=16, batch_size=8, n_epochs=1, hidden_layers=[16, 16])
            config['training'].update(evaluation_episodes=1, evaluate_every_episodes=2)
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            run = train(3, path, Path(directory) / 'run')
            manifest = json.loads((run / 'run.json').read_text())
            self.assertEqual(manifest['completed_episodes'], 3)
            self.assertEqual(manifest['training_steps'], 60)
            self.assertGreater(manifest['optimizer_epochs'], 0)
            with (run / 'episodes.csv').open() as output:
                self.assertEqual(len(list(csv.DictReader(output))), 3)
            self.assertTrue((run / 'best.zip').exists())
            self.assertTrue((run / 'best_normalization.pkl').exists())
            self.assertTrue(callable(saved_reward(run)))


class RestoredBaselineTests(unittest.TestCase):
    def test_matches_previous_dense_reward_and_dynamics(self):
        from David_trying_stuff.train_walking import WalkingEnv
        config = load_config()
        config['rewards'].update(fall_penalty=0., finish_bonus=0., forward_reach_weight=0.)
        old = WalkingEnv()
        new = RunningEnv(config)
        try:
            old.reset(seed=123)
            new.reset(seed=123)
            self.assertEqual(new.observation_space.shape, (348,))
            np.testing.assert_allclose(new.data.qpos, old.data.qpos)
            for _ in range(20):
                action = np.full(17, .02)
                _, old_reward, old_done, _, _ = old.step(action)
                _, new_reward, new_done, _, _ = new.step(action)
                np.testing.assert_allclose(old.data.qpos, new.data.qpos, atol=1e-10)
                self.assertAlmostEqual(old_reward, new_reward, places=8)
                self.assertEqual(old_done, new_done)
                if old_done:
                    break
        finally:
            old.close()
            new.close()

    def test_python_variables_and_optional_phase_observation(self):
        config = load_config()
        self.assertEqual(config['ppo']['gamma'], .95)
        self.assertEqual(config['ppo']['activation'], 'relu')
        config['rewards']['knee_weight'] = 1.
        env = RunningEnv(config)
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (369,))
        finally:
            env.close()


class ForwardReachTests(unittest.TestCase):
    def test_reach_requires_upright_forward_motion_and_swing(self):
        from .rewards import forward_reach_reward
        weights = load_config()['rewards']
        foot = dict(ahead=.35, clearance=.1, relative_vx=1., contact=False)
        state = dict(fell=False, torso_up=1., height=1.4, vx=1., heading=1., swing_feet=[foot, foot])
        self.assertAlmostEqual(forward_reach_reward(state, weights), .5)
        for change in (dict(fell=True), dict(torso_up=.7), dict(vx=0.), dict(vx=-1.), dict(heading=-1.)):
            self.assertEqual(forward_reach_reward(dict(state, **change), weights), 0.)
        for change in (dict(contact=True), dict(clearance=.01), dict(relative_vx=0.), dict(ahead=-.1)):
            feet = [dict(foot, **change), dict(foot, **change)]
            self.assertEqual(forward_reach_reward(dict(state, swing_feet=feet), weights), 0.)

    def test_reach_caps_and_treats_both_sides_equally(self):
        from .rewards import forward_reach_reward
        weights = load_config()['rewards']
        swing = dict(ahead=.35, clearance=.1, relative_vx=1., contact=False)
        support = dict(swing, contact=True)
        state = dict(fell=False, torso_up=1., height=1.4, vx=1., heading=1., swing_feet=[swing, support])
        expected = forward_reach_reward(state, weights)
        self.assertAlmostEqual(expected, .25)
        self.assertEqual(forward_reach_reward(dict(state, swing_feet=[support, swing]), weights), expected)
        self.assertEqual(forward_reach_reward(dict(state, swing_feet=[dict(swing, ahead=2.), support]), weights), expected)


if __name__ == '__main__':
    unittest.main()
