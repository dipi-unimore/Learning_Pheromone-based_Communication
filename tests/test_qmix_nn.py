import os
import tempfile
import unittest

import numpy as np
import torch

from agents.QMIXLearningNN import qmix_nn


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class DummyLogger:
    def __init__(self):
        self.values = []

    def load_value(self, value):
        self.values.append(value)

    def empty_table(self):
        pass

    def save_computation_time(self, *_args, **_kwargs):
        pass


class DummySlimeEnv:
    def __init__(self):
        self.cluster_learners = 1
        self.scatter_learners = 0
        self.learners = {0: {"mode": "c", "pos": (0, 0)}}
        self.W = 1
        self.H = 1
        self.patch_size = 1
        self.patches = []
        self.fov = []
        self.ph_fov = []
        self._step = 0
        self._state = 0

    def actions_n(self):
        return 2

    def observations_n(self):
        return 3

    def convert_observation(self, obs):
        return int(np.asarray(obs)[0])

    def reset(self):
        self._step = 0

    def agent_iter(self, max_iter):
        yield "0"

    def last(self, agent):
        reward = 1.0 if self._step == 0 else 0.5
        return np.array([self._state]), reward, False, False, {}

    def step(self, action):
        self._step += 1
        self._state = (self._state + 1) % 3

    def avg_cluster(self):
        return 0.0

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_flat(module: torch.nn.Module) -> torch.Tensor:
    """Concatenate all parameters of a module into a single flat tensor."""
    return torch.cat([p.detach().view(-1) for p in module.parameters()])


_DEFAULT_L_PARAMS = {
    "alpha": 0.01,
    "gamma": 0.9,
    "epsilon": 0.0,
    "epsilon_min": 0.0,
    "decay_type": "log",
    "decay": 0.99,
    "train_episodes": 1,
    "test_episodes": 1,
    "agent_hidden_dim": 4,
    "mixer_hidden_dim": 4,
    "mixer_learning_rate": 0.01,
    "device": "cpu",
    "replay_capacity": 128,
    "batch_size": 4,
    "learning_starts": 4,
    "train_every": 1,
    "target_update_mode": "hard",
    "target_update_interval": 10,
    "tau": 0.01,
}

_DEFAULT_PARAMS = {
    "cluster_learners": 1,
    "scatter_learners": 0,
    "actions": ["a0", "a1"],
    "episode_ticks": 1,
}


# ---------------------------------------------------------------------------
# Unit tests — neural-network modules
# ---------------------------------------------------------------------------

class TestAgentQNetwork(unittest.TestCase):
    def _make_net(self):
        return qmix_nn.AgentQNetwork(
            obs_dim=3, hidden_dim=4, n_actions=2,
            learning_rate=0.05, device=torch.device("cpu"),
        )

    def test_predict_returns_numpy_of_correct_shape(self):
        net = self._make_net()
        q = net.predict(np.array([0.0, 1.0, 0.0]))
        self.assertIsInstance(q, np.ndarray)
        self.assertEqual(q.shape, (2,))

    def test_update_td_moves_q_toward_target(self):
        """After a TD update the target action's Q-value should increase."""
        net = self._make_net()
        state = np.array([0.0, 1.0, 0.0])
        before = net.predict(state).copy()
        net.update_td(state, action=1, target=before[1] + 1.0)
        after = net.predict(state)
        self.assertGreater(after[1], before[1])

    def test_update_td_changes_parameters(self):
        net = self._make_net()
        state = np.array([0.0, 1.0, 0.0])
        before = _params_flat(net).clone()
        net.update_td(state, action=0, target=5.0)
        after = _params_flat(net)
        self.assertFalse(torch.allclose(before, after))


class TestQMIXMixer(unittest.TestCase):
    def _make_mixer(self, n_agents=1):
        return qmix_nn.QMIXMixer(
            n_agents=n_agents, state_dim=3 * n_agents, hidden_dim=4,
            learning_rate=0.05, device=torch.device("cpu"),
        )

    def test_predict_returns_float(self):
        mixer = self._make_mixer()
        gs = np.array([0.0, 1.0, 0.0])
        q_total = mixer.predict(gs, np.array([1.5]))
        self.assertIsInstance(q_total, float)

    def test_mixing_weights_are_non_negative(self):
        """Softplus ensures no weight is negative — IGM property."""
        mixer = self._make_mixer(n_agents=2)
        gs = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        gs_t = torch.tensor(gs, dtype=torch.float32)
        with torch.no_grad():
            weights = torch.nn.functional.softplus(mixer.hyper_w(gs_t))
        self.assertTrue((weights >= 0).all())

    def test_update_td_changes_parameters(self):
        mixer = self._make_mixer()
        gs = np.array([0.0, 1.0, 0.0])
        aqs = np.array([1.0])
        before = _params_flat(mixer).clone()
        mixer.update_td(gs, aqs, team_reward=1.0, q_total_next=0.0, gamma=0.9)
        after = _params_flat(mixer)
        self.assertFalse(torch.allclose(before, after))

    def test_multi_agent_predict(self):
        mixer = self._make_mixer(n_agents=3)
        gs = np.zeros(9, dtype=np.float32)
        aqs = np.array([0.5, 0.8, 1.2])
        q_total = mixer.predict(gs, aqs)
        self.assertIsInstance(q_total, float)


class TestReplayBuffer(unittest.TestCase):
    def test_replay_capacity_is_enforced(self):
        replay = qmix_nn.ReplayBuffer(capacity=2)
        for i in range(3):
            replay.push(
                qmix_nn.Transition(
                    global_state=np.array([i], dtype=np.float32),
                    actions=np.array([0], dtype=np.int64),
                    team_reward=1.0,
                    next_global_state=np.array([i + 1], dtype=np.float32),
                    done=0.0,
                )
            )
        self.assertEqual(len(replay), 2)

    def test_replay_sample_batch_size(self):
        replay = qmix_nn.ReplayBuffer(capacity=10)
        for i in range(5):
            replay.push(
                qmix_nn.Transition(
                    global_state=np.array([i], dtype=np.float32),
                    actions=np.array([0], dtype=np.int64),
                    team_reward=1.0,
                    next_global_state=np.array([i + 1], dtype=np.float32),
                    done=0.0,
                )
            )
        batch = replay.sample(3)
        self.assertEqual(len(batch), 3)
        self.assertTrue(all(isinstance(x, qmix_nn.Transition) for x in batch))


# ---------------------------------------------------------------------------
# Unit tests — create_agent
# ---------------------------------------------------------------------------

class TestCreateAgent(unittest.TestCase):
    def test_train_tuple_length_and_types(self):
        result = qmix_nn.create_agent(_DEFAULT_PARAMS, _DEFAULT_L_PARAMS, 3, 2, train=True)
        self.assertEqual(len(result), 18)
        agent_nets, mixer, device, agent_lr, mixer_lr, gamma, *_ = result
        self.assertEqual(len(agent_nets), 1)
        self.assertIsInstance(mixer, qmix_nn.QMIXMixer)
        self.assertIsInstance(device, torch.device)
        self.assertAlmostEqual(agent_lr, 0.01)
        self.assertAlmostEqual(mixer_lr, 0.01)
        self.assertAlmostEqual(gamma, 0.9)

    def test_eval_tuple_length_and_types(self):
        result = qmix_nn.create_agent(_DEFAULT_PARAMS, _DEFAULT_L_PARAMS, 3, 2, train=False)
        self.assertEqual(len(result), 11)
        agent_nets, mixer, device, episodes, *_ = result
        self.assertEqual(len(agent_nets), 1)
        self.assertIsInstance(device, torch.device)
        self.assertEqual(episodes, 1)

    def test_network_dimensions(self):
        agent_nets, mixer, device, *_ = qmix_nn.create_agent(
            _DEFAULT_PARAMS, _DEFAULT_L_PARAMS, 3, 2, train=True
        )
        self.assertEqual(agent_nets[0].obs_dim, 3)
        self.assertEqual(agent_nets[0].n_actions, 2)
        self.assertEqual(mixer.n_agents, 1)
        self.assertEqual(mixer.state_dim, 3)  # 1 agent × 3 obs

    def test_device_cpu_forced(self):
        _, _, device, *_ = qmix_nn.create_agent(
            _DEFAULT_PARAMS, {**_DEFAULT_L_PARAMS, "device": "cpu"}, 3, 2, train=True
        )
        self.assertEqual(str(device), "cpu")


# ---------------------------------------------------------------------------
# Unit tests — serialization
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):
    def test_save_load_round_trip_preserves_predictions(self):
        agent_nets, mixer, device, *_ = qmix_nn.create_agent(
            _DEFAULT_PARAMS, _DEFAULT_L_PARAMS, 3, 2, train=True
        )
        state = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gs = np.concatenate([state])
        q_before = agent_nets[0].predict(state).copy()
        qtotal_before = mixer.predict(gs, np.array([q_before[0]], dtype=np.float32))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        try:
            qmix_nn.save_model(agent_nets, mixer, tmp)
            # Load into fresh networks
            new_nets, new_mixer, new_dev, *_ = qmix_nn.create_agent(
                _DEFAULT_PARAMS, _DEFAULT_L_PARAMS, 3, 2, train=True
            )
            qmix_nn.load_model(new_nets, new_mixer, tmp, new_dev)

            q_after = new_nets[0].predict(state)
            qtotal_after = new_mixer.predict(gs, np.array([q_after[0]], dtype=np.float32))

            np.testing.assert_allclose(q_before, q_after, rtol=1e-5)
            self.assertAlmostEqual(qtotal_before, qtotal_after, places=5)
        finally:
            os.unlink(tmp)

    def test_checkpoint_has_expected_keys(self):
        agent_nets, mixer, *_ = qmix_nn.create_agent(
            _DEFAULT_PARAMS, _DEFAULT_L_PARAMS, 3, 2, train=True
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        try:
            qmix_nn.save_model(agent_nets, mixer, tmp)
            ckpt = torch.load(tmp, map_location="cpu")
            self.assertIn("agent_nets", ckpt)
            self.assertIn("mixer", ckpt)
            self.assertEqual(len(ckpt["agent_nets"]), 1)
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Integration smoke tests
# ---------------------------------------------------------------------------

class TestTrainEvalSmoke(unittest.TestCase):
    def setUp(self):
        self.env = DummySlimeEnv()
        self.logger = DummyLogger()

    def _create_train_args(self):
        return qmix_nn.create_agent(_DEFAULT_PARAMS, _DEFAULT_L_PARAMS, 3, 2, train=True)

    def test_train_smoke(self):
        (
            agent_nets, mixer, device,
            agent_lr, mixer_lr, gamma, epsilon, epsilon_min, decay_type, decay,
            episodes,
            cluster_dict, cluster_actions_dict, cluster_action_dict, cluster_reward_dict,
            scatter_actions_dict, scatter_action_dict, scatter_reward_dict,
        ) = self._create_train_args()

        trained_nets, trained_mixer = qmix_nn.train(
            self.env, _DEFAULT_PARAMS, _DEFAULT_L_PARAMS,
            agent_nets, mixer,
            cluster_dict, cluster_actions_dict, cluster_action_dict, cluster_reward_dict,
            scatter_actions_dict, scatter_action_dict, scatter_reward_dict,
            episodes, 1, agent_lr, mixer_lr, gamma, decay_type, decay,
            epsilon, epsilon_min, 1, self.logger, None,
        )

        self.assertEqual(len(trained_nets), 1)
        self.assertIsInstance(trained_mixer, qmix_nn.QMIXMixer)
        self.assertTrue(self.logger.values, "logger should have received at least one row")

    def test_no_update_before_learning_starts(self):
        l_params = {**_DEFAULT_L_PARAMS, "learning_starts": 1000, "batch_size": 2}
        (
            agent_nets,
            mixer,
            device,
            agent_lr,
            mixer_lr,
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
        ) = qmix_nn.create_agent(_DEFAULT_PARAMS, l_params, 3, 2, train=True)

        before = torch.cat([_params_flat(agent_nets[0]), _params_flat(mixer)]).clone()
        qmix_nn.train(
            self.env,
            _DEFAULT_PARAMS,
            l_params,
            agent_nets,
            mixer,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
            episodes,
            1,
            agent_lr,
            mixer_lr,
            gamma,
            decay_type,
            decay,
            epsilon,
            epsilon_min,
            1,
            self.logger,
            None,
        )
        after = torch.cat([_params_flat(agent_nets[0]), _params_flat(mixer)])
        self.assertTrue(torch.allclose(before, after))

    def test_update_after_learning_starts(self):
        params = {**_DEFAULT_PARAMS, "episode_ticks": 4}
        l_params = {
            **_DEFAULT_L_PARAMS,
            "learning_starts": 1,
            "batch_size": 1,
            "train_every": 1,
            "target_update_interval": 1,
        }
        (
            agent_nets,
            mixer,
            device,
            agent_lr,
            mixer_lr,
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
        ) = qmix_nn.create_agent(params, l_params, 3, 2, train=True)

        before = torch.cat([_params_flat(agent_nets[0]), _params_flat(mixer)]).clone()
        qmix_nn.train(
            self.env,
            params,
            l_params,
            agent_nets,
            mixer,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
            episodes,
            1,
            agent_lr,
            mixer_lr,
            gamma,
            decay_type,
            decay,
            epsilon,
            epsilon_min,
            1,
            self.logger,
            None,
        )
        after = torch.cat([_params_flat(agent_nets[0]), _params_flat(mixer)])
        self.assertFalse(torch.allclose(before, after))

    def test_eval_smoke(self):
        # First train briefly to get networks
        agent_nets, mixer, device, *_ = self._create_train_args()

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        try:
            qmix_nn.save_model(agent_nets, mixer, tmp)

            (
                eval_nets, eval_mixer, eval_dev,
                test_episodes,
                cluster_dict, cluster_actions_dict, cluster_action_dict, cluster_reward_dict,
                scatter_actions_dict, scatter_action_dict, scatter_reward_dict,
            ) = qmix_nn.create_agent(_DEFAULT_PARAMS, _DEFAULT_L_PARAMS, 3, 2, train=False)
            qmix_nn.load_model(eval_nets, eval_mixer, tmp, eval_dev)

            qmix_nn.eval(
                self.env, _DEFAULT_PARAMS,
                cluster_dict, cluster_actions_dict, cluster_action_dict, cluster_reward_dict,
                scatter_actions_dict, scatter_action_dict, scatter_reward_dict,
                test_episodes, eval_nets, 1, self.logger, None,
            )
        finally:
            os.unlink(tmp)

        self.assertTrue(self.logger.values)


if __name__ == "__main__":
    unittest.main()
