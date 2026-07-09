import importlib.util
import random
from pathlib import Path

import numpy as np
import torch

# import agents.QMIXLearningNN.qmix_nn as qmix_nn

# _SPEC = importlib.util.spec_from_file_location("qmix_nn", Path(__file__).with_name("qmix_nn.py"))
_SPEC = importlib.util.spec_from_file_location("qmix_nn", Path("agents/QMIXLearningNN/qmix_nn.py"))
qmix_nn = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(qmix_nn)


def _base_params(cluster_learners=2, scatter_learners=2):
    return {
        "cluster_learners": cluster_learners,
        "scatter_learners": scatter_learners,
    }


def _base_l_params():
    return {
        "alpha": 0.01,
        "gamma": 0.9,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "decay_type": "log",
        "decay": 0.99,
        "train_episodes": 1,
        "test_episodes": 1,
        "agent_hidden_dim": 4,
        "mixer_hidden_dim": 5,
        "mixer_learning_rate": 0.01,
        "device": "cpu",
    }


def _clone_params(module):
    return [p.detach().clone() for p in module.parameters()]


def _params_equal(before, module):
    return all(torch.equal(old, new.detach()) for old, new in zip(before, module.parameters()))


def _params_changed(before, module):
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, module.parameters()))


def test_create_agent_builds_independent_mode_mixers():
    created = qmix_nn.create_agent(
        _base_params(cluster_learners=2, scatter_learners=3),
        _base_l_params(),
        n_obs=5,
        n_actions=4,
        train=True,
    )

    agent_nets, mixers = created[0], created[1]

    assert len(agent_nets) == 5
    assert isinstance(mixers, dict)
    assert set(mixers.keys()) == {"c", "s"}
    assert mixers["c"].n_agents == 2
    assert mixers["c"].state_dim == 2 * 5
    assert mixers["s"].n_agents == 3
    assert mixers["s"].state_dim == 3 * 5
    assert mixers["c"] is not mixers["s"]


def test_sample_and_update_only_steps_same_mode_agents():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cpu")
    n_obs = 3
    n_actions = 2
    agent_nets = [qmix_nn.AgentQNetwork(n_obs, 4, n_actions, 0.01, device) for _ in range(3)]
    target_agent_nets = [qmix_nn.AgentQNetwork(n_obs, 4, n_actions, 0.01, device) for _ in range(3)]
    for i in range(3):
        qmix_nn._hard_update(target_agent_nets[i], agent_nets[i])

    mixer_c = qmix_nn.QMIXMixer(n_agents=2, state_dim=2 * n_obs, hidden_dim=5, learning_rate=0.01, device=device)
    target_mixer_c = qmix_nn.QMIXMixer(n_agents=2, state_dim=2 * n_obs, hidden_dim=5, learning_rate=0.01, device=device)
    qmix_nn._hard_update(target_mixer_c, mixer_c)

    replay = qmix_nn.ReplayBuffer(capacity=4)
    replay.push(
        qmix_nn.Transition(
            global_state=np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32),
            actions=np.asarray([0, 1], dtype=np.int64),
            team_reward=1.0,
            next_global_state=np.asarray([0, 1, 0, 0, 0, 1], dtype=np.float32),
            done=0.0,
        )
    )
    replay.push(
        qmix_nn.Transition(
            global_state=np.asarray([0, 1, 0, 1, 0, 0], dtype=np.float32),
            actions=np.asarray([1, 0], dtype=np.int64),
            team_reward=-0.5,
            next_global_state=np.asarray([0, 0, 1, 0, 1, 0], dtype=np.float32),
            done=0.0,
        )
    )

    cluster_0_before = _clone_params(agent_nets[0])
    cluster_1_before = _clone_params(agent_nets[1])
    scatter_before = _clone_params(agent_nets[2])
    mixer_before = _clone_params(mixer_c)

    qmix_nn._sample_and_update(
        replay=replay,
        batch_size=2,
        agent_ids=[0, 1],
        n_obs=n_obs,
        gamma=0.9,
        device=device,
        agent_nets=agent_nets,
        mixer=mixer_c,
        target_agent_nets=target_agent_nets,
        target_mixer=target_mixer_c,
    )

    assert _params_changed(cluster_0_before, agent_nets[0]) or _params_changed(cluster_1_before, agent_nets[1])
    assert _params_changed(mixer_before, mixer_c)
    assert _params_equal(scatter_before, agent_nets[2])


def test_pack_unpack_round_trips_mode_specific_mixers():
    torch.manual_seed(123)
    np.random.seed(123)

    params = _base_params(cluster_learners=1, scatter_learners=1)
    l_params = _base_l_params()
    agent_nets, mixers = qmix_nn.create_agent(params, l_params, n_obs=3, n_actions=2, train=False)[:2]
    packed = qmix_nn.pack_model(agent_nets, mixers)

    restored_agent_nets, restored_mixers = qmix_nn.create_agent(params, l_params, n_obs=3, n_actions=2, train=False)[:2]
    qmix_nn.unpack_model(packed, restored_agent_nets, restored_mixers)

    for source, restored in zip(agent_nets, restored_agent_nets):
        for source_param, restored_param in zip(source.parameters(), restored.parameters()):
            assert torch.allclose(source_param, restored_param)

    for mode in ["c", "s"]:
        for source_param, restored_param in zip(mixers[mode].parameters(), restored_mixers[mode].parameters()):
            assert torch.allclose(source_param, restored_param)


class _FakeLogger:
    def __init__(self):
        self.rows = []
        self.emptied = False

    def load_value(self, value):
        self.rows.append(value)

    def empty_table(self):
        self.emptied = True


class _TinyTwoModeEnv:
    def __init__(self):
        self.cluster_learners = 1
        self.scatter_learners = 1
        self.learners = {0: {"mode": "c"}, 1: {"mode": "s"}}
        self.closed = False
        self._obs_counter = 0

    def observations_n(self):
        return 3

    def actions_n(self):
        return 2

    def reset(self):
        self._obs_counter = 0

    def agent_iter(self, max_iter):
        for agent_id in range(max_iter):
            yield str(agent_id)

    def last(self, agent):
        agent_id = int(agent)
        obs = np.zeros(self.observations_n(), dtype=np.float32)
        obs[(self._obs_counter + agent_id) % self.observations_n()] = 1.0
        reward = 1.0 if self.learners[agent_id]["mode"] == "c" else -1.0
        self._obs_counter += 1
        return obs, reward, False, False, {}

    def convert_observation(self, obs):
        return int(np.argmax(obs))

    def step(self, action):
        assert 0 <= int(action) < self.actions_n()

    def avg_cluster(self):
        return 1.0, 1.0, 1.0, 1.0

    def close(self):
        self.closed = True


def test_train_smoke_uses_mode_specific_replay_and_mixers():
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    env = _TinyTwoModeEnv()
    params = {
        "cluster_learners": 1,
        "scatter_learners": 1,
        "episode_ticks": 3,
    }
    l_params = _base_l_params()
    l_params.update(
        {
            "train_episodes": 1,
            "replay_capacity": 8,
            "batch_size": 1,
            "learning_starts": 1,
            "train_every": 1,
            "target_update_interval": 1,
            "target_update_mode": "hard",
        }
    )
    created = qmix_nn.create_agent(params, l_params, n_obs=env.observations_n(), n_actions=env.actions_n(), train=True)
    (
        agent_nets,
        mixers,
        _device,
        agent_lr,
        mixer_lr,
        gamma,
        epsilon,
        epsilon_min,
        decay_type,
        decay,
        train_episodes,
        cluster_dict,
        cluster_actions_dict,
        cluster_action_dict,
        cluster_reward_dict,
        scatter_actions_dict,
        scatter_action_dict,
        scatter_reward_dict,
    ) = created
    logger = _FakeLogger()

    returned_agent_nets, returned_mixers = qmix_nn.train(
        env=env,
        params=params,
        l_params=l_params,
        agent_nets=agent_nets,
        mixer=mixers,
        cluster_dict=cluster_dict,
        cluster_actions_dict=cluster_actions_dict,
        cluster_action_dict=cluster_action_dict,
        cluster_reward_dict=cluster_reward_dict,
        scatter_actions_dict=scatter_actions_dict,
        scatter_action_dict=scatter_action_dict,
        scatter_reward_dict=scatter_reward_dict,
        train_episodes=train_episodes,
        train_log_every=1,
        agent_learning_rate=agent_lr,
        mixer_learning_rate=mixer_lr,
        gamma=gamma,
        decay_type=decay_type,
        decay=decay,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        print_metrics=999,
        logger=logger,
        visualizer=None,
    )

    assert returned_agent_nets is agent_nets
    assert returned_mixers is mixers
    assert set(returned_mixers.keys()) == {"c", "s"}
    assert logger.rows
    assert logger.emptied
    assert env.closed
