import unittest

import numpy as np

from go1_t3.control import JOINT_MAX, STAND, position_command
from go1_t3.safety import HOLD_TIMEOUT, SafetyBlock, SelfCollision


class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collision = SelfCollision()

    def test_safe_request_is_applied_in_steps_of_at_most_five_degrees(self):
        block = SafetyBlock(self.collision)
        q = STAND.copy()
        q[[0, 3]] += [0.22, -0.22]
        self.assertTrue(block.receive(position_command(q, 45, 1), 1.0))
        previous = STAND
        for tick in range(4):
            target = block.step(1.0 + tick * 0.01)[:, 0].copy()
            self.assertLessEqual(np.max(np.abs(target - previous)), np.deg2rad(5))
            previous = target
        np.testing.assert_allclose(previous, q)

    def test_recognizes_a_forbidden_collision_inside_joint_limits(self):
        # The published pair 31,39 intersects by about 3.46 mm.
        q = np.array([-0.848812, -0.322350, -2.145124, 0.749517, 2.909521, -1.436219,
                      -0.475363, 3.574871, -1.727294, -0.307991, 3.817975, -2.402089])
        self.assertFalse(self.collision.collides(STAND))
        self.assertTrue(self.collision.collides(q))
        block = SafetyBlock(self.collision)
        block.receive(position_command(q, 45, 1), 0.0)
        for tick in range(35):
            target = block.step(tick * 0.01)[:, 0]
            self.assertFalse(self.collision.collides(target))
        self.assertGreater(np.max(np.abs(target - q)), 0.01)

    def test_checks_between_collision_free_endpoints(self):
        start = np.array([0.06476, 2.54965, -1.00699, 0.37850, -0.04367, -1.97872,
                          -0.62998, 1.31045, -2.55689, -0.40781, 3.77548, -0.98796])
        target = np.array([-0.33476, 1.02872, -1.10659, 0.83451, -0.14600, -0.94438,
                           -0.11700, 0.83023, -2.18970, 0.31561, 0.14543, -2.72231])
        self.assertFalse(self.collision.collides(start))
        self.assertFalse(self.collision.collides(target))
        self.assertTrue(self.collision.collides((start + target) / 2))
        self.assertFalse(self.collision.transition_is_safe(start, target))

    def test_invalid_input_preserves_output_and_timeout_holds_target(self):
        block = SafetyBlock(self.collision)
        live = position_command(STAND, 45, 1)
        live[:, 1:3] = 0.2
        self.assertTrue(block.receive(live, 1.0))
        accepted = block.step(1.01).copy()
        for field in range(5):
            for invalid in (np.nan, np.inf, -np.inf):
                bad = live.copy()
                bad[11, field] = invalid
                self.assertFalse(block.receive(bad, 1.1))
                np.testing.assert_array_equal(block.step(1.11), accepted)
        held = block.step(1.001 + HOLD_TIMEOUT)
        np.testing.assert_array_equal(held[:, 0], accepted[:, 0])
        np.testing.assert_array_equal(held[:, 1:3], 0)
        self.assertTrue(np.all(held[:, 3:] > 0))

    def test_clips_joint_limit_before_publishing(self):
        block = SafetyBlock(self.collision)
        q = STAND.copy()
        q[0] = JOINT_MAX[0] + 0.5
        block.receive(position_command(q, 45, 1), 0.0)
        for tick in range(12):
            output = block.step(tick * 0.01)
            self.assertLessEqual(output[0, 0], JOINT_MAX[0])
        self.assertGreater(output[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
