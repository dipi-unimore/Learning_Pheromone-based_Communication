import unittest
from types import SimpleNamespace

import numpy as np

from agents.CoQLearning.coql import (
    apply_collaboration,
    resolve_collaboration_config,
    should_share_information,
)


class TestCoQL(unittest.TestCase):
    def test_resolve_collaboration_config_defaults_to_full_sharing(self):
        collaboration = resolve_collaboration_config({})

        self.assertTrue(collaboration["enabled"])
        self.assertEqual(collaboration["share_every_steps"], 1)
        self.assertEqual(collaboration["recipient_selector"], "all")
        self.assertEqual(
            collaboration["shared_information"],
            {
                "observation": True,
                "action": True,
                "reward": True,
                "q_values": True,
            },
        )

    def test_should_share_information_respects_step_interval(self):
        collaboration = resolve_collaboration_config(
            {
                "collaboration": {
                    "share_every_steps": 2,
                    "shared_information": {"reward": True},
                }
            }
        )

        self.assertFalse(should_share_information(collaboration, 1))
        self.assertTrue(should_share_information(collaboration, 2))

    def test_apply_collaboration_with_similar_selector_uses_shared_q_values(self):
        env = SimpleNamespace(W=5, H=5, patch_size=1, learners={0: {"mode": "c"}, 1: {"mode": "c"}})
        qtable = np.zeros((2, 3, 2), dtype=float)
        qtable[1, 2] = np.array([6.0, 8.0])
        step_info = {
            0: {
                "state": 0,
                "observation": np.array([1, 0, 1]),
                "reward": 1.0,
                "current_action": 0,
                "previous_action": None,
                "position": (0, 0),
            },
            1: {
                "state": 2,
                "observation": np.array([1, 0, 1]),
                "reward": 2.0,
                "current_action": 1,
                "previous_action": 0,
                "position": (4, 4),
            },
        }
        collaboration = resolve_collaboration_config(
            {
                "collaboration": {
                    "recipient_selector": "similar",
                    "share_rate": 0.25,
                    "shared_information": {
                        "observation": False,
                        "action": False,
                        "reward": False,
                        "q_values": True,
                    },
                }
            }
        )

        updated_qtable = apply_collaboration(qtable, env, step_info, collaboration, alpha=0.5, gamma=0.9)

        np.testing.assert_allclose(updated_qtable[0, 0], np.array([1.5, 2.0]))

    def test_apply_collaboration_observation_shares_peer_observed_states(self):
        env = SimpleNamespace(W=5, H=5, patch_size=1, learners={0: {"mode": "c"}, 1: {"mode": "c"}})
        qtable = np.zeros((2, 3, 2), dtype=float)
        qtable[0, 2] = np.array([2.0, 4.0])
        step_info = {
            0: {
                "state": 0,
                "observation": np.array([1, 0, 0]),
                "reward": 0.0,
                "current_action": 0,
                "previous_action": None,
                "position": (0, 0),
            },
            1: {
                "state": 2,
                "observation": np.array([0, 1, 0]),
                "reward": 0.0,
                "current_action": 1,
                "previous_action": 0,
                "position": (1, 1),
            },
        }
        collaboration = resolve_collaboration_config(
            {
                "collaboration": {
                    "share_rate": 0.5,
                    "shared_information": {
                        "observation": True,
                        "action": False,
                        "reward": False,
                        "q_values": False,
                    },
                }
            }
        )

        updated_qtable = apply_collaboration(qtable, env, step_info, collaboration, alpha=0.5, gamma=0.9)

        np.testing.assert_allclose(updated_qtable[0, 0], np.array([1.0, 2.0]))

    def test_apply_collaboration_reward_shares_average_peer_reward(self):
        env = SimpleNamespace(W=5, H=5, patch_size=1, learners={0: {"mode": "c"}, 1: {"mode": "c"}})
        qtable = np.zeros((2, 2, 2), dtype=float)
        step_info = {
            0: {
                "state": 1,
                "observation": np.array([1, 0]),
                "reward": 1.0,
                "current_action": 0,
                "previous_action": None,
                "position": (0, 0),
            },
            1: {
                "state": 0,
                "observation": np.array([0, 1]),
                "reward": 4.0,
                "current_action": 1,
                "previous_action": 0,
                "position": (1, 1),
            },
        }
        collaboration = resolve_collaboration_config(
            {
                "collaboration": {
                    "share_rate": 1.0,
                    "shared_information": {
                        "observation": False,
                        "action": False,
                        "reward": True,
                        "q_values": False,
                    },
                }
            }
        )

        updated_qtable = apply_collaboration(qtable, env, step_info, collaboration, alpha=1.0, gamma=0.0)

        self.assertEqual(updated_qtable[0, 1, 0], 4.0)

    def test_apply_collaboration_nearby_selector_uses_wrapped_distance(self):
        env = SimpleNamespace(
            W=5,
            H=5,
            patch_size=1,
            learners={0: {"mode": "c"}, 1: {"mode": "c"}, 2: {"mode": "c"}},
        )
        qtable = np.zeros((3, 2, 2), dtype=float)
        step_info = {
            0: {
                "state": 0,
                "observation": np.array([1, 0]),
                "reward": 0.0,
                "current_action": 0,
                "previous_action": None,
                "position": (0, 0),
            },
            1: {
                "state": 1,
                "observation": np.array([0, 1]),
                "reward": 0.0,
                "current_action": 1,
                "previous_action": 1,
                "position": (4, 0),
            },
            2: {
                "state": 1,
                "observation": np.array([0, 1]),
                "reward": 0.0,
                "current_action": 1,
                "previous_action": 1,
                "position": (2, 2),
            },
        }
        collaboration = resolve_collaboration_config(
            {
                "collaboration": {
                    "recipient_selector": "nearby",
                    "nearby_radius": 1,
                    "share_rate": 0.5,
                    "shared_information": {
                        "observation": False,
                        "action": True,
                        "reward": False,
                        "q_values": False,
                    },
                }
            }
        )

        updated_qtable = apply_collaboration(qtable, env, step_info, collaboration, alpha=0.5, gamma=0.9)

        self.assertEqual(updated_qtable[0, 0, 1], 0.5)
        self.assertEqual(updated_qtable[2, 1, 0], 0.0)
        self.assertEqual(updated_qtable[2, 1, 1], 0.0)

    def test_apply_collaboration_filters_to_same_agent_kind(self):
        env = SimpleNamespace(
            W=5,
            H=5,
            patch_size=1,
            learners={0: {"mode": "c"}, 1: {"mode": "s"}},
        )
        qtable = np.zeros((2, 2, 2), dtype=float)
        qtable[1, 1] = np.array([8.0, 8.0])
        step_info = {
            0: {
                "state": 0,
                "observation": np.array([1, 0]),
                "reward": 0.0,
                "current_action": 0,
                "previous_action": None,
                "position": (0, 0),
            },
            1: {
                "state": 1,
                "observation": np.array([1, 0]),
                "reward": 5.0,
                "current_action": 1,
                "previous_action": 1,
                "position": (1, 1),
            },
        }
        collaboration = resolve_collaboration_config(
            {
                "collaboration": {
                    "recipient_selector": "all",
                    "share_rate": 1.0,
                    "shared_information": {
                        "observation": False,
                        "action": False,
                        "reward": True,
                        "q_values": False,
                    },
                }
            }
        )

        updated_qtable = apply_collaboration(qtable, env, step_info, collaboration, alpha=1.0, gamma=0.0)

        self.assertEqual(updated_qtable[0, 0, 0], 0.0)
        self.assertEqual(updated_qtable[0, 0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()


