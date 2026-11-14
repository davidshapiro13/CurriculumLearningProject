"""Run with: python -m unittest discover -s LearningMujoco -p 'test_*.py'."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from train_walking import WalkingEnv


class WalkingTests(unittest.TestCase):
    def setUp(self):
        self.env = WalkingEnv()
        self.env.reset(seed=0)
        self.addCleanup(self.env.close)

    def test_original_model(self):
        self.assertEqual(self.env.model.nu, 17)
        self.assertEqual(self.env.observation_space.shape, (348,))
        self.assertEqual(self.env.model.neq, 0)
        self.assertTrue(np.all(self.env.model.actuator_biasprm == 0))

    def test_contact_pairs_and_margin(self):
        floor = self.env.floor_id
        foot = self.env.model.geom('left_foot').id
        hand = self.env.model.geom('right_hand').id
        for pair, distance, expected in [
            ((floor, foot), -0.001, None), ((foot, floor), 0, None),
            ((floor, hand), 0, 'right_lower_arm'),
            ((hand, floor), -0.001, 'right_lower_arm'),
            ((floor, hand), 0.0001, None), ((foot, hand), -0.01, None),
        ]:
            contact = SimpleNamespace(geom1=pair[0], geom2=pair[1], dist=distance)
            with patch.object(self.env, 'data', SimpleNamespace(contact=[contact], ncon=1)):
                self.assertEqual(self.env.fallen_body(), expected)

    def test_real_fall_terminates(self):
        for _ in range(1000):
            _, _, terminated, truncated, info = self.env.step(np.zeros(self.env.model.nu))
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIsNotNone(info['fallen_body'])
        with self.assertRaises(RuntimeError):
            self.env.step(np.zeros(self.env.model.nu))

    def test_exact_time_limit_without_ground_contact(self):
        self.env.model.opt.gravity[:] = 0
        self.env.data.qpos[2] = 100  # Keep the floor out of reach for this test.
        mujoco.mj_forward(self.env.model, self.env.data)
        for step in range(1, 1001):
            _, _, terminated, truncated, info = self.env.step(np.zeros(self.env.model.nu))
            self.assertFalse(terminated)
            self.assertEqual(truncated, step == 1000)
        self.assertEqual(info['end_reason'], 'time_limit')
        self.env.reset(seed=0)
        self.assertEqual(self.env.elapsed_steps, 0)

    def test_substep_contact_stops_immediately(self):
        # Contact on the second physics substep must not be lost by later motion.
        with patch.object(self.env, 'fallen_body', side_effect=[None, None, 'left_shin']):
            _, _, terminated, truncated, _ = self.env.step(np.zeros(self.env.model.nu))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(self.env.data.time, 2 * self.env.model.opt.timestep)

    def test_fall_at_time_limit_is_termination(self):
        self.env.elapsed_steps = 999
        with patch.object(self.env, 'fallen_body', return_value='left_shin'):
            _, _, terminated, truncated, info = self.env.step(np.zeros(self.env.model.nu))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['end_reason'], 'fall')

    def test_height_does_not_replace_contact_termination(self):
        self.env.data.qpos[2] = 3.0
        mujoco.mj_forward(self.env.model, self.env.data)
        _, _, terminated, truncated, info = self.env.step(np.zeros(self.env.model.nu))
        self.assertFalse(terminated or truncated)
        self.assertEqual(info['upright_reward'], 0.0)

    def test_action_limits_and_reset_seed(self):
        first, _ = self.env.reset(seed=42)
        second, _ = self.env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        for value, expected in [(1, self.env.upper), (-1, self.env.lower), (0, self.env.neutral)]:
            self.env.reset(seed=0)
            self.env.step(np.full(self.env.model.nu, value))
            np.testing.assert_allclose(self.env.data.ctrl, expected)


if __name__ == '__main__':
    unittest.main()
