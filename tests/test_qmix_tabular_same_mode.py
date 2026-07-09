import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location("qmix", Path(__file__).with_name("qmix.py"))
qmix = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(qmix)


class SpyMixingNetwork(qmix.MixingNetwork):
    def __init__(self, n_agents: int, n_actions: int):
        super().__init__(n_agents=n_agents, n_actions=n_actions, learning_rate=0.0)
        self.weights.fill(0.0)
        self.value.fill(0.0)
        self.forward_calls = []
        self.backward_calls = []

    def forward(self, individual_q_values, action_idx):
        self.forward_calls.append((tuple(float(v) for v in individual_q_values), int(action_idx)))
        return super().forward(individual_q_values, action_idx)

    def backward(self, individual_q_values, action_idx, td_error):
        self.backward_calls.append(
            (tuple(float(v) for v in individual_q_values), int(action_idx), float(td_error))
        )
        return super().backward(individual_q_values, action_idx, td_error)


class DummyLogger:
    def __init__(self):
        self.values = []
        self.emptied = False

    def load_value(self, value):
        self.values.append(value)

    def empty_table(self):
        self.emptied = True


class FakeSlimeEnv:
    def __init__(self):
        self.cluster_learners = 2
        self.scatter_learners = 1
        self.learners = {
            0: {"mode": "c", "pos": (0, 0)},
            1: {"mode": "c", "pos": (1, 0)},
            2: {"mode": "s", "pos": (2, 0)},
        }
        self.patches = {}
        self.fov = {}
        self.ph_fov = {}
        self.closed = False
        self._tick = 0
        self._actions = []

    def actions_n(self):
        return 2

    def observations_n(self):
        return 3

    def reset(self):
        self._tick = 0
        self._actions.clear()

    def agent_iter(self, max_iter):
        self._tick += 1
        for agent_id in range(max_iter):
            yield str(agent_id)

    def last(self, agent):
        agent_id = int(agent)
        obs = np.asarray([(self._tick + agent_id) % self.observations_n()], dtype=np.float32)
        reward = {0: 1.0, 1: 2.0, 2: 100.0}[agent_id]
        return obs, reward, False, False, {}

    def convert_observation(self, obs):
        return int(np.asarray(obs).reshape(-1)[0])

    def step(self, action):
        self._actions.append(int(action))

    def avg_cluster(self):
        return 0.0, 0.0, 0.0, 0.0

    def close(self):
        self.closed = True


def _learning_params():
    return {
        "alpha": 0.1,
        "gamma": 0.0,
        "epsilon": 0.0,
        "epsilon_min": 0.0,
        "decay_type": "linear",
        "decay": 1.0,
        "train_episodes": 1,
        "test_episodes": 1,
        "mixing_learning_rate": 0.0,
    }


def test_create_agent_builds_independent_mode_mixers():
    params = {"cluster_learners": 2, "scatter_learners": 3}
    created = qmix.create_agent(params, _learning_params(), n_obs=5, n_actions=4, train=True)

    qtable, mixers = created[:2]

    assert qtable.shape == (5, 5, 4)
    assert set(mixers) == {"c", "s"}
    assert mixers["c"].n_agents == 2
    assert mixers["s"].n_agents == 3
    assert mixers["c"] is not mixers["s"]


def test_mixing_network_rejects_wrong_number_of_agent_q_values():
    mixer = qmix.MixingNetwork(n_agents=2, n_actions=2)

    with pytest.raises(ValueError, match="expected 2 individual Q-values"):
        mixer.forward([1.0, 2.0, 3.0], 0)

    with pytest.raises(ValueError, match="expected 2 individual Q-values"):
        mixer.backward([1.0], 0, 1.0)


def test_legacy_all_agent_mixer_is_rejected_when_both_modes_exist():
    legacy_all_agent_mixer = qmix.MixingNetwork(n_agents=4, n_actions=2)

    with pytest.raises(ValueError, match="same-mode constraint"):
        qmix._ensure_mode_mixing_networks(
            legacy_all_agent_mixer,
            {"c": [0, 1], "s": [2, 3]},
        )


def test_update_mode_mixing_networks_uses_only_same_mode_data():
    qtable = np.zeros((4, 3, 2), dtype=float)
    qtable[0, 0, 0] = 10.0
    qtable[1, 1, 1] = 20.0
    qtable[2, 0, 0] = 100.0
    qtable[3, 1, 1] = 200.0

    old_s = {"0": 0, "1": 1, "2": 0, "3": 1}
    old_a = {"0": 0, "1": 1, "2": 0, "3": 1}
    step_info = {
        0: {"state": 2, "reward": 1.0},
        1: {"state": 2, "reward": 2.0},
        2: {"state": 2, "reward": 100.0},
        3: {"state": 2, "reward": 200.0},
    }
    mixers = {"c": SpyMixingNetwork(2, 2), "s": SpyMixingNetwork(2, 2)}

    qmix._update_mode_mixing_networks(
        qtable=qtable,
        mixing_nets=mixers,
        step_info=step_info,
        old_s=old_s,
        old_a=old_a,
        agent_ids_by_mode={"c": [0, 1], "s": [2, 3]},
        n_actions=2,
        gamma=0.0,
    )

    assert [call[0] for call in mixers["c"].backward_calls] == [(10.0, 20.0), (10.0, 20.0)]
    assert [call[2] for call in mixers["c"].backward_calls] == [3.0, 3.0]
    assert [call[0] for call in mixers["s"].backward_calls] == [(100.0, 200.0), (100.0, 200.0)]
    assert [call[2] for call in mixers["s"].backward_calls] == [300.0, 300.0]


def test_train_smoke_updates_mode_specific_mixers_only():
    env = FakeSlimeEnv()
    params = {"cluster_learners": 2, "scatter_learners": 1, "episode_ticks": 2}
    l_params = _learning_params()
    qtable = np.zeros((3, env.observations_n(), env.actions_n()), dtype=float)
    mixers = {"c": SpyMixingNetwork(2, env.actions_n()), "s": SpyMixingNetwork(1, env.actions_n())}
    logger = DummyLogger()

    returned_qtable, returned_mixers = qmix.train(
        env=env,
        params=params,
        l_params=l_params,
        qtable=qtable,
        mixing_net=mixers,
        cluster_dict={"1": 0.0},
        cluster_actions_dict={"1": {"0": 0, "1": 0}},
        cluster_action_dict={"1": {"0": {"0": 0, "1": 0}, "1": {"0": 0, "1": 0}}},
        cluster_reward_dict={"1": {"0": 0, "1": 0}},
        scatter_actions_dict={"1": {"0": 0, "1": 0}},
        scatter_action_dict={"1": {"2": {"0": 0, "1": 0}}},
        scatter_reward_dict={"1": {"2": 0}},
        train_episodes=1,
        train_log_every=1,
        alpha=0.1,
        gamma=0.0,
        decay_type="linear",
        decay=1.0,
        epsilon=0.0,
        epsilon_min=0.0,
        print_metrics=999,
        logger=logger,
        visualizer=None,
    )

    assert returned_qtable is qtable
    assert returned_mixers["c"] is mixers["c"]
    assert returned_mixers["s"] is mixers["s"]
    assert mixers["c"].backward_calls
    assert mixers["s"].backward_calls
    assert all(len(call[0]) == 2 for call in mixers["c"].backward_calls)
    assert all(len(call[0]) == 1 for call in mixers["s"].backward_calls)
    assert logger.emptied is True
    assert env.closed is True
