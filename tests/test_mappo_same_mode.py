import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
for candidate in [TEST_DIR, REPO_ROOT / "agents" / "MAPPOLearning", REPO_ROOT]:
    sys.path.insert(0, str(candidate))

try:
    import mappo  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - used when tests run from repo root
    from agents.MAPPOLearning import mappo  # type: ignore  # noqa: E402


def _base_params(cluster=2, scatter=2):
    return {"cluster_learners": cluster, "scatter_learners": scatter, "episode_ticks": 3}


def _base_l_params():
    return {
        "alpha": 0.003,
        "gamma": 0.9,
        "train_episodes": 2,
        "test_episodes": 1,
        "device": "cpu",
        "actor_hidden_dim": 8,
        "critic_hidden_dim": 8,
        "actor_learning_rate": 0.003,
        "critic_learning_rate": 0.003,
        "actor_parameter_sharing": "none",
        "rollout_steps": 2,
        "ppo_epochs": 2,
        "minibatch_size": 2,
        "clip_ratio": 0.2,
        "gae_lambda": 0.95,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "normalize_advantages": False,
        "evaluation_policy": "deterministic",
    }


def _flat_params(module):
    return torch.cat([p.detach().cpu().reshape(-1) for p in module.parameters()])


def _module_changed(before, module):
    return not torch.allclose(before, _flat_params(module))


def _push_rollout(buffer, actor_nets, agent_ids, obs_dim=3, n_actions=2, n_steps=4):
    rng = np.random.default_rng(123)
    for step in range(n_steps):
        global_state = rng.normal(size=len(agent_ids) * obs_dim).astype(np.float32)
        next_global_state = rng.normal(size=len(agent_ids) * obs_dim).astype(np.float32)
        actions = []
        old_log_probs = []
        for local_i, agent_id in enumerate(agent_ids):
            obs = global_state[local_i * obs_dim : (local_i + 1) * obs_dim]
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=actor_nets[agent_id].device).unsqueeze(0)
            dist = actor_nets[agent_id].distribution(obs_t)
            action = int((step + local_i) % n_actions)
            actions.append(action)
            old_log_probs.append(float(dist.log_prob(torch.tensor([action], device=actor_nets[agent_id].device)).item()))
        buffer.push(
            mappo.ModeTransition(
                global_state=global_state,
                actions=np.asarray(actions, dtype=np.int64),
                old_log_probs=np.asarray(old_log_probs, dtype=np.float32),
                team_reward=float(1.0 + step),
                next_global_state=next_global_state,
                done=float(step == n_steps - 1),
            )
        )


def test_create_agent_builds_mode_specific_critics():
    torch.manual_seed(1)
    params = _base_params(cluster=2, scatter=1)
    l_params = _base_l_params()

    created = mappo.create_agent(params, l_params, n_obs=3, n_actions=2, train=True)
    actor_nets, critics = created[0], created[1]

    assert len(actor_nets) == 3
    assert set(critics.keys()) == {"c", "s"}
    assert critics["c"].state_dim == 2 * 3
    assert critics["s"].state_dim == 1 * 3


def test_same_mode_actor_parameter_sharing_does_not_cross_modes():
    params = _base_params(cluster=2, scatter=2)
    l_params = _base_l_params()
    l_params["actor_parameter_sharing"] = "same_mode"

    actor_nets, critics = mappo.create_agent(params, l_params, n_obs=3, n_actions=2, train=True)[:2]

    assert actor_nets[0] is actor_nets[1]
    assert actor_nets[2] is actor_nets[3]
    assert actor_nets[0] is not actor_nets[2]
    assert set(critics.keys()) == {"c", "s"}


def test_ppo_update_only_steps_same_mode_modules():
    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)

    params = _base_params(cluster=2, scatter=2)
    l_params = _base_l_params()
    actor_nets, critics = mappo.create_agent(params, l_params, n_obs=3, n_actions=2, train=True)[:2]
    buffer = mappo.ModeRolloutBuffer()
    _push_rollout(buffer, actor_nets, agent_ids=[0, 1], obs_dim=3, n_actions=2, n_steps=5)

    actor_before = [_flat_params(actor).clone() for actor in actor_nets]
    critic_c_before = _flat_params(critics["c"]).clone()
    critic_s_before = _flat_params(critics["s"]).clone()

    mappo._ppo_update_mode(
        buffer=buffer,
        agent_ids=[0, 1],
        obs_dim=3,
        actor_nets=actor_nets,
        critic=critics["c"],
        gamma=0.9,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=2,
        minibatch_size=2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        normalize_advantages=False,
    )

    assert _module_changed(actor_before[0], actor_nets[0])
    assert _module_changed(actor_before[1], actor_nets[1])
    assert not _module_changed(actor_before[2], actor_nets[2])
    assert not _module_changed(actor_before[3], actor_nets[3])
    assert _module_changed(critic_c_before, critics["c"])
    assert not _module_changed(critic_s_before, critics["s"])


def test_pack_unpack_round_trips_mode_specific_models():
    torch.manual_seed(5)
    params = _base_params(cluster=1, scatter=1)
    l_params = _base_l_params()

    actor_nets, critics = mappo.create_agent(params, l_params, n_obs=3, n_actions=2, train=True)[:2]
    packed = mappo.pack_model(actor_nets, critics)

    restored_actor_nets, restored_critics = mappo.create_agent(params, l_params, n_obs=3, n_actions=2, train=True)[:2]
    mappo.unpack_model(packed, restored_actor_nets, restored_critics)

    for source, restored in zip(actor_nets, restored_actor_nets):
        assert torch.allclose(_flat_params(source), _flat_params(restored))
    for mode in ["c", "s"]:
        assert torch.allclose(_flat_params(critics[mode]), _flat_params(restored_critics[mode]))


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
        self.actions_taken = []

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

    def step(self, action):
        assert 0 <= int(action) < self.actions_n()
        self.actions_taken.append(int(action))

    def avg_cluster(self):
        return 1.0, 1.0, 1.0, 1.0

    def close(self):
        self.closed = True


def test_train_smoke_uses_same_mode_rollouts_and_closes_env():
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    env = _TinyTwoModeEnv()
    params = {"cluster_learners": 1, "scatter_learners": 1, "episode_ticks": 3}
    l_params = _base_l_params()
    l_params.update({"train_episodes": 1, "rollout_steps": 1, "ppo_epochs": 1, "minibatch_size": 1})

    created = mappo.create_agent(params, l_params, n_obs=env.observations_n(), n_actions=env.actions_n(), train=True)
    (
        actor_nets,
        critics,
        _device,
        actor_lr,
        critic_lr,
        gamma,
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
    returned_actor_nets, returned_critics = mappo.train(
        env=env,
        params=params,
        l_params=l_params,
        actor_nets=actor_nets,
        critics=critics,
        cluster_dict=cluster_dict,
        cluster_actions_dict=cluster_actions_dict,
        cluster_action_dict=cluster_action_dict,
        cluster_reward_dict=cluster_reward_dict,
        scatter_actions_dict=scatter_actions_dict,
        scatter_action_dict=scatter_action_dict,
        scatter_reward_dict=scatter_reward_dict,
        train_episodes=train_episodes,
        train_log_every=1,
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        gamma=gamma,
        print_metrics=999,
        logger=logger,
        visualizer=None,
    )

    assert returned_actor_nets is actor_nets
    assert returned_critics is critics
    assert set(returned_critics.keys()) == {"c", "s"}
    assert logger.rows
    assert logger.emptied
    assert env.closed

class _ScriptedEvalActor:
    def greedy_action(self, obs_vec):
        del obs_vec
        return 0

    def act(self, obs_vec):
        del obs_vec
        return 1, 0.0, 0.0


def test_resolve_evaluation_policy_accepts_canonical_values_and_aliases():
    assert mappo.resolve_evaluation_policy({}) == "deterministic"
    assert mappo.resolve_evaluation_policy({"evaluation_policy": "deterministic"}) == "deterministic"
    assert mappo.resolve_evaluation_policy({"evaluation_policy": "greedy"}) == "deterministic"
    assert mappo.resolve_evaluation_policy({"evaluation_policy": "argmax"}) == "deterministic"
    assert mappo.resolve_evaluation_policy({"evaluation_policy": "stochastic"}) == "stochastic"
    assert mappo.resolve_evaluation_policy({"evaluation_policy": "sample"}) == "stochastic"
    assert mappo.resolve_evaluation_policy({"evaluation_action_selection": "categorical"}) == "stochastic"

    with pytest.raises(ValueError):
        mappo.resolve_evaluation_policy({"evaluation_policy": "epsilon_greedy"})


def _run_eval_with_scripted_policy(policy):
    env = _TinyTwoModeEnv()
    params = {"cluster_learners": 1, "scatter_learners": 1, "episode_ticks": 2}
    l_params = _base_l_params()
    l_params.update({"test_episodes": 1, "evaluation_policy": policy})

    created = mappo.create_agent(params, l_params, n_obs=env.observations_n(), n_actions=env.actions_n(), train=False)
    (
        _actor_nets,
        _critics,
        _device,
        test_episodes,
        cluster_dict,
        cluster_actions_dict,
        cluster_action_dict,
        cluster_reward_dict,
        scatter_actions_dict,
        scatter_action_dict,
        scatter_reward_dict,
    ) = created

    logger = _FakeLogger()
    mappo.eval(
        env=env,
        params=params,
        cluster_dict=cluster_dict,
        cluster_actions_dict=cluster_actions_dict,
        cluster_action_dict=cluster_action_dict,
        cluster_reward_dict=cluster_reward_dict,
        scatter_actions_dict=scatter_actions_dict,
        scatter_action_dict=scatter_action_dict,
        scatter_reward_dict=scatter_reward_dict,
        test_episodes=test_episodes,
        actor_nets=[_ScriptedEvalActor(), _ScriptedEvalActor()],
        test_log_every=1,
        logger=logger,
        visualizer=None,
        l_params=l_params,
    )
    assert env.closed
    assert logger.emptied
    return env.actions_taken


def test_eval_uses_configured_deterministic_or_stochastic_policy():
    assert _run_eval_with_scripted_policy("deterministic") == [0, 0, 0, 0]
    assert _run_eval_with_scripted_policy("stochastic") == [1, 1, 1, 1]

