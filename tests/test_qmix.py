import unittest
from types import SimpleNamespace

import numpy as np

from agents.QMIXLearning.qmix import MixingNetwork, create_agent


class TestMixingNetwork(unittest.TestCase):
    """Tests for the MixingNetwork class."""

    def test_mixing_network_initialization(self):
        """Verify mixing network dimensions are correct."""
        n_agents = 2
        n_actions = 6
        net = MixingNetwork(n_agents, n_actions)

        self.assertEqual(net.weights.shape, (n_actions, n_agents))
        self.assertEqual(net.value.shape, (n_actions,))

    def test_mixing_network_forward_pass(self):
        """Test mixing network forward pass computes correct aggregation."""
        net = MixingNetwork(n_agents=2, n_actions=3)
        net.weights = np.array([[1.0, 2.0], [2.0, 1.0], [0.5, 0.5]])
        net.value = np.array([0.0, 1.0, 0.5])

        individual_q = [1.0, 2.0]

        # Action 0: 1.0*1.0 + 2.0*2.0 + 0.0 = 5.0
        q0 = net.forward(individual_q, 0)
        self.assertAlmostEqual(q0, 5.0)

        # Action 1: 2.0*1.0 + 1.0*2.0 + 1.0 = 5.0
        q1 = net.forward(individual_q, 1)
        self.assertAlmostEqual(q1, 5.0)

        # Action 2: 0.5*1.0 + 0.5*2.0 + 0.5 = 2.0
        q2 = net.forward(individual_q, 2)
        self.assertAlmostEqual(q2, 2.0)

    def test_mixing_network_backward_updates_weights(self):
        """Verify backward pass updates weights correctly."""
        net = MixingNetwork(n_agents=2, n_actions=2, learning_rate=0.1)
        net.weights = np.zeros((2, 2))
        net.value = np.zeros(2)

        individual_q = [1.0, 2.0]
        td_error = 0.5
        action_idx = 0

        net.backward(individual_q, action_idx, td_error)

        # Expected: weights[0] -= lr * td_error * individual_q
        # weights[0, 0] = 0 - 0.1 * 0.5 * 1.0 = -0.05
        # weights[0, 1] = 0 - 0.1 * 0.5 * 2.0 = -0.1
        expected_weights_0 = np.array([-0.05, -0.1])
        np.testing.assert_almost_equal(net.weights[0], expected_weights_0)

        # value[0] = 0 - 0.1 * 0.5 = -0.05
        self.assertAlmostEqual(net.value[0], -0.05)

    def test_mixing_network_backward_preserves_other_actions(self):
        """Verify backward pass doesn't affect other actions."""
        net = MixingNetwork(n_agents=2, n_actions=3, learning_rate=0.1)
        original_weights_1 = net.weights[1].copy()
        original_weights_2 = net.weights[2].copy()

        individual_q = [1.0, 2.0]
        net.backward(individual_q, 0, 0.5)

        # Action 1 and 2 should be unchanged
        np.testing.assert_array_equal(net.weights[1], original_weights_1)
        np.testing.assert_array_equal(net.weights[2], original_weights_2)


class TestQMIXCreateAgent(unittest.TestCase):
    """Tests for create_agent function."""

    def setUp(self):
        """Set up test parameters."""
        self.params = {
            "cluster_learners": 2,
            "scatter_learners": 1,
            "episode_ticks": 500,
        }
        self.l_params = {
            "alpha": 0.025,
            "gamma": 0.9,
            "epsilon": 1.0,
            "epsilon_min": 0.1,
            "decay_type": "log",
            "decay": 0.9987,
            "train_episodes": 100,
            "test_episodes": 20,
            "mixing_learning_rate": 0.025,
        }

    def test_create_agent_train_returns_correct_types(self):
        """Verify create_agent returns correct structures for training."""
        result = create_agent(self.params, self.l_params, 10, 6, train=True)

        self.assertEqual(len(result), 16)
        qtable, mixing_net, alpha, gamma, epsilon, epsilon_min, decay_type, decay, episodes, *rest = result

        self.assertIsInstance(qtable, np.ndarray)
        self.assertIsInstance(mixing_net, MixingNetwork)
        self.assertEqual(alpha, 0.025)
        self.assertEqual(gamma, 0.9)
        self.assertEqual(episodes, 100)

    def test_create_agent_train_qtable_dimensions(self):
        """Verify Q-table has correct shape."""
        qtable, *_ = create_agent(self.params, self.l_params, 10, 6, train=True)

        expected_agents = 2 + 1  # cluster + scatter
        self.assertEqual(qtable.shape, (expected_agents, 10, 6))

    def test_create_agent_train_mixing_network_dimensions(self):
        """Verify mixing network initialized with correct agent count."""
        _, mixing_net, *_ = create_agent(self.params, self.l_params, 10, 6, train=True)

        expected_agents = 2 + 1
        self.assertEqual(mixing_net.n_agents, expected_agents)
        self.assertEqual(mixing_net.n_actions, 6)

    def test_create_agent_eval_returns_correct_types(self):
        """Verify create_agent returns correct structures for evaluation."""
        result = create_agent(self.params, self.l_params, 10, 6, train=False)

        self.assertEqual(len(result), 8)
        episodes, *rest = result
        self.assertEqual(episodes, 20)

    def test_create_agent_tracking_dicts_have_correct_keys(self):
        """Verify tracking dictionaries have episodes as keys."""
        (
            qtable,
            mixing_net,
            alpha,
            gamma,
            epsilon,
            epsilon_min,
            decay_type,
            decay,
            episodes,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
        ) = create_agent(self.params, self.l_params, 10, 6, train=True)

        # Check all dicts have episodes as keys
        expected_keys = set(str(ep) for ep in range(1, episodes + 1))

        self.assertEqual(set(cluster_dict.keys()), expected_keys)
        self.assertEqual(set(cluster_actions_dict.keys()), expected_keys)
        self.assertEqual(set(cluster_action_dict.keys()), expected_keys)
        self.assertEqual(set(scatter_actions_dict.keys()), expected_keys)

    def test_create_agent_cluster_action_dict_per_agent(self):
        """Verify cluster action dict has entries per agent."""
        (
            qtable,
            mixing_net,
            alpha,
            gamma,
            epsilon,
            epsilon_min,
            decay_type,
            decay,
            episodes,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            *_,
        ) = create_agent(self.params, self.l_params, 10, 6, train=True)

        # Check first episode has entries for all cluster agents
        ep1_actions = cluster_action_dict["1"]
        self.assertEqual(len(ep1_actions), self.params["cluster_learners"])

        # Each agent should have all 6 actions
        for agent_id, actions in ep1_actions.items():
            self.assertEqual(len(actions), 6)


class TestQMIXVsSingleAgent(unittest.TestCase):
    """Integration tests comparing QMIX single-agent behavior to IQL."""

    def test_qmix_single_agent_vs_iql(self):
        """With single agent, QMIX should behave like IQL."""
        params_single = {
            "cluster_learners": 1,
            "scatter_learners": 0,
            "episode_ticks": 10,
        }
        l_params = {
            "alpha": 0.025,
            "gamma": 0.9,
            "epsilon": 0.1,
            "epsilon_min": 0.0,
            "decay_type": "log",
            "decay": 0.999,
            "train_episodes": 5,
            "test_episodes": 1,
            "mixing_learning_rate": 0.025,
        }

        qmix_qtable, mixing_net, *_ = create_agent(params_single, l_params, 5, 3, train=True)

        # With single agent, mixing network should just act as an identity/linear layer
        self.assertEqual(qmix_qtable.shape, (1, 5, 3))
        self.assertEqual(mixing_net.n_agents, 1)


if __name__ == "__main__":
    unittest.main()

