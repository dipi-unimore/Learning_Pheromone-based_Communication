"""
MAPPO for the Slime environment, implemented with PyTorch.

This module follows the same runner-facing style as the existing learning
modules while enforcing the Slime same-mode learning constraint:

* cluster learners (mode "c") train with cluster learners only;
* scatter learners (mode "s") train with scatter learners only.

The implementation uses decentralized actors and mode-specific centralized
critics. A mode critic sees the concatenated observations of agents of that
mode only. PPO advantages are computed per mode from same-mode rewards, and
actor updates are restricted to agents in that same mode.
"""

import itertools
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

def _align_logger_row_with_schema(value: List[float], logger, label: str) -> List[float]:
    metrics = getattr(logger, "metrics", None)
    if metrics is None:
        return value

    expected = len(metrics)
    actual = len(value)
    if actual == expected:
        return value

    if actual + 1 == expected:
        missing_metric = str(metrics[-1]).lower()
        if "epsilon" in missing_metric or missing_metric in {"eps", "exploration", "exploration_rate"}:
            value.append(float("nan"))
            return value

    raise ValueError(
        f"MAPPO logger row/schema mismatch during {label}: "
        f"row has {actual} values but logger expects {expected} columns. "
        "This usually means the MAPPO metric row and Logger.metrics schema are out of sync."
    )

# ---------------------------------------------------------------------------
# Device / observation helpers
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _as_obs_vec(obs, obs_dim: int) -> NDArray:
    """Convert a Slime observation to a fixed-size float32 vector."""

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_dim = int(obs_dim)

    if arr.size == obs_dim:
        return arr.astype(np.float32, copy=False)

    # Robust fallback for tests or discrete observations: treat scalar integer
    # observations as one-hot ids when the target dimension is larger than one.
    if arr.size == 1 and obs_dim > 1:
        out = np.zeros(obs_dim, dtype=np.float32)
        idx = int(arr.item())
        if 0 <= idx < obs_dim:
            out[idx] = 1.0
        else:
            out[idx % obs_dim] = 1.0
        return out

    out = np.zeros(obs_dim, dtype=np.float32)
    n = min(obs_dim, arr.size)
    out[:n] = arr[:n]
    return out


def _agent_ids_by_mode_from_counts(cluster_learners: int, scatter_learners: int) -> Dict[str, List[int]]:
    ids_by_mode: Dict[str, List[int]] = {}
    if int(cluster_learners) > 0:
        ids_by_mode["c"] = list(range(int(cluster_learners)))
    if int(scatter_learners) > 0:
        start = int(cluster_learners)
        ids_by_mode["s"] = list(range(start, start + int(scatter_learners)))
    return ids_by_mode


def _agent_ids_by_mode_from_env(env) -> Dict[str, List[int]]:
    ids_by_mode: Dict[str, List[int]] = {}
    n_agents = int(env.cluster_learners + env.scatter_learners)
    for agent_id in range(n_agents):
        mode = env.learners[agent_id]["mode"]
        ids_by_mode.setdefault(mode, []).append(agent_id)
    return ids_by_mode


def _unique_modules(modules: Iterable[nn.Module]) -> List[nn.Module]:
    seen = set()
    unique = []
    for module in modules:
        key = id(module)
        if key not in seen:
            seen.add(key)
            unique.append(module)
    return unique


# ---------------------------------------------------------------------------
# Neural modules
# ---------------------------------------------------------------------------


class ActorPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int, learning_rate: float, device: torch.device) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), self.n_actions),
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=float(learning_rate))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))

    def act(self, obs_vec: NDArray) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.distribution(obs_t)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)
            entropy_t = dist.entropy()
        return int(action_t.item()), float(log_prob_t.item()), float(entropy_t.item())

    def greedy_action(self, obs_vec: NDArray) -> int:
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(obs_t)
            return int(torch.argmax(logits, dim=-1).item())

    def sample_action(self, obs_vec: NDArray) -> int:
        action, _log_prob, _entropy = self.act(obs_vec)
        return int(action)


_EVALUATION_POLICY_ALIASES = {
    "deterministic": "deterministic",
    "greedy": "deterministic",
    "argmax": "deterministic",
    "max": "deterministic",
    "mode": "deterministic",
    "stochastic": "stochastic",
    "sample": "stochastic",
    "sampling": "stochastic",
    "categorical": "stochastic",
    "policy_sample": "stochastic",
}


def resolve_evaluation_policy(l_params: Optional[Dict] = None, evaluation_policy: Optional[str] = None) -> str:
    """Return the normalized MAPPO evaluation policy.

    Supported canonical values are:
    - ``"deterministic"``: choose ``argmax`` over actor logits.
    - ``"stochastic"``: sample from the actor's categorical policy.

    ``l_params`` may use the preferred key ``evaluation_policy``. The older
    aliases ``eval_policy`` and ``evaluation_action_selection`` are accepted to
    make experiment files easier to read and migrate.
    """

    if evaluation_policy is not None:
        raw_policy = evaluation_policy
    elif l_params is None:
        raw_policy = "deterministic"
    elif "evaluation_policy" in l_params:
        raw_policy = l_params["evaluation_policy"]
    elif "eval_policy" in l_params:
        raw_policy = l_params["eval_policy"]
    elif "evaluation_action_selection" in l_params:
        raw_policy = l_params["evaluation_action_selection"]
    elif "eval_action_selection" in l_params:
        raw_policy = l_params["eval_action_selection"]
    elif "deterministic_eval" in l_params:
        raw_policy = bool(l_params["deterministic_eval"])
    else:
        raw_policy = "deterministic"

    if isinstance(raw_policy, bool):
        return "deterministic" if raw_policy else "stochastic"

    key = str(raw_policy).strip().lower().replace("-", "_")
    if key not in _EVALUATION_POLICY_ALIASES:
        valid = ", ".join(sorted({"deterministic", "stochastic"}))
        aliases = ", ".join(sorted(_EVALUATION_POLICY_ALIASES))
        raise ValueError(f"evaluation_policy must be one of: {valid}. Accepted aliases: {aliases}")
    return _EVALUATION_POLICY_ALIASES[key]


def resolve_evaluation_action_selection(l_params: Optional[Dict] = None) -> str:
    """Compatibility wrapper for evaluation action-selection config."""

    return resolve_evaluation_policy(l_params=l_params)


def _select_evaluation_action(actor: ActorPolicy, obs_vec: NDArray, evaluation_policy: str) -> int:
    policy = resolve_evaluation_policy(evaluation_policy=evaluation_policy)
    if policy == "deterministic":
        return int(actor.greedy_action(obs_vec))
    action, _log_prob, _entropy = actor.act(obs_vec)
    return int(action)


class CentralValueCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, learning_rate: float, device: torch.device) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=float(learning_rate))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


@dataclass
class ModeTransition:
    global_state: NDArray
    actions: NDArray
    old_log_probs: NDArray
    team_reward: float
    next_global_state: NDArray
    done: float


class ModeRolloutBuffer:
    def __init__(self) -> None:
        self._data: List[ModeTransition] = []

    def __len__(self) -> int:
        return len(self._data)

    def push(self, transition: ModeTransition) -> None:
        self._data.append(transition)

    def clear(self) -> None:
        self._data.clear()

    def as_list(self) -> List[ModeTransition]:
        return list(self._data)


# ---------------------------------------------------------------------------
# Model construction / validation
# ---------------------------------------------------------------------------


def _create_actor_nets(
    agent_ids_by_mode: Dict[str, List[int]],
    n_agents: int,
    obs_dim: int,
    hidden_dim: int,
    n_actions: int,
    learning_rate: float,
    device: torch.device,
    parameter_sharing: str,
) -> List[ActorPolicy]:
    if parameter_sharing not in {"none", "same_mode"}:
        raise ValueError("actor_parameter_sharing must be one of: none, same_mode")

    actor_nets: List[ActorPolicy] = [None] * int(n_agents)  # type: ignore[list-item]
    if parameter_sharing == "same_mode":
        for mode, agent_ids in agent_ids_by_mode.items():
            del mode
            shared_actor = ActorPolicy(obs_dim, hidden_dim, n_actions, learning_rate, device)
            for agent_id in agent_ids:
                actor_nets[agent_id] = shared_actor
    else:
        for agent_id in range(int(n_agents)):
            actor_nets[agent_id] = ActorPolicy(obs_dim, hidden_dim, n_actions, learning_rate, device)
    return actor_nets


def _create_mode_critics(
    agent_ids_by_mode: Dict[str, List[int]],
    obs_dim: int,
    hidden_dim: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, CentralValueCritic]:
    critics: Dict[str, CentralValueCritic] = {}
    for mode, agent_ids in agent_ids_by_mode.items():
        if agent_ids:
            critics[mode] = CentralValueCritic(
                state_dim=len(agent_ids) * int(obs_dim),
                hidden_dim=int(hidden_dim),
                learning_rate=float(learning_rate),
                device=device,
            )
    return critics


def _assert_mode_critics(critics: Dict[str, CentralValueCritic], agent_ids_by_mode: Dict[str, List[int]], obs_dim: int) -> None:
    expected_modes = {mode for mode, ids in agent_ids_by_mode.items() if ids}
    actual_modes = set(critics.keys())
    if actual_modes != expected_modes:
        raise ValueError(f"MAPPO critics must be mode-specific. Expected modes {sorted(expected_modes)}, got {sorted(actual_modes)}")
    for mode, ids in agent_ids_by_mode.items():
        if not ids:
            continue
        expected_state_dim = len(ids) * int(obs_dim)
        if critics[mode].state_dim != expected_state_dim:
            raise ValueError(
                f"critic for mode {mode!r} has state_dim={critics[mode].state_dim}, expected {expected_state_dim}"
            )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _module_param_vector(module: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().cpu().view(-1) for p in module.parameters()]).numpy().astype(np.float32)


def _copy_param_vector_to_module(param_vec: NDArray, module: nn.Module, device: torch.device) -> None:
    vec = np.asarray(param_vec, dtype=np.float32).ravel()
    idx = 0
    for p in module.parameters():
        n = p.numel()
        if idx + n > vec.size:
            raise ValueError("flat parameter vector ended before all module parameters were restored")
        chunk = torch.from_numpy(vec[idx : idx + n]).to(device=device, dtype=p.dtype).view_as(p)
        with torch.no_grad():
            p.copy_(chunk)
        idx += n
    if idx != vec.size:
        raise ValueError("flat parameter vector contains unused values")


def pack_model(actor_nets: List[ActorPolicy], critics: Dict[str, CentralValueCritic]) -> NDArray:
    """Pack actor and critic parameters into a flat float32 vector for Logger .npy storage."""

    flat_parts: List[np.ndarray] = []
    flat_parts.append(np.asarray([float(len(actor_nets))], dtype=np.float32))
    for actor in actor_nets:
        params = _module_param_vector(actor)
        flat_parts.append(np.asarray([float(params.size)], dtype=np.float32))
        flat_parts.append(params)

    ordered_modes = [mode for mode in ("c", "s") if mode in critics]
    flat_parts.append(np.asarray([float(len(ordered_modes))], dtype=np.float32))
    for mode in ordered_modes:
        params = _module_param_vector(critics[mode])
        flat_parts.append(np.asarray([float(ord(mode))], dtype=np.float32))
        flat_parts.append(np.asarray([float(params.size)], dtype=np.float32))
        flat_parts.append(params)

    return np.concatenate(flat_parts).astype(np.float32)


def unpack_model(flat_weights: NDArray, actor_nets: List[ActorPolicy], critics: Dict[str, CentralValueCritic]) -> None:
    """Restore a model produced by pack_model()."""

    vec = np.asarray(flat_weights, dtype=np.float32).ravel()
    offset = 0
    if vec.size == 0:
        raise ValueError("empty MAPPO model vector")

    n_actors = int(vec[offset])
    offset += 1
    if n_actors != len(actor_nets):
        raise ValueError(f"model has {n_actors} actors, current MAPPO agent has {len(actor_nets)}")

    for actor in actor_nets:
        count = int(vec[offset])
        offset += 1
        actor_vec = vec[offset : offset + count]
        offset += count
        _copy_param_vector_to_module(actor_vec, actor, actor.device)

    n_critics = int(vec[offset])
    offset += 1
    for _ in range(n_critics):
        mode = chr(int(vec[offset]))
        offset += 1
        count = int(vec[offset])
        offset += 1
        critic_vec = vec[offset : offset + count]
        offset += count
        if mode not in critics:
            raise ValueError(f"model contains critic for unavailable mode {mode!r}")
        _copy_param_vector_to_module(critic_vec, critics[mode], critics[mode].device)

    if offset != vec.size:
        raise ValueError("MAPPO model vector contains trailing unused values")


def save_model(actor_nets: List[ActorPolicy], critics: Dict[str, CentralValueCritic], path: str) -> None:
    torch.save(
        {
            "actor_nets": [actor.state_dict() for actor in actor_nets],
            "critics": {mode: critic.state_dict() for mode, critic in critics.items()},
        },
        path,
    )


def load_model(actor_nets: List[ActorPolicy], critics: Dict[str, CentralValueCritic], path: str, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    for i, state_dict in enumerate(checkpoint["actor_nets"]):
        actor_nets[i].load_state_dict(state_dict)
    for mode, state_dict in checkpoint["critics"].items():
        critics[mode].load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Tracking helpers copied to keep launcher-independent API compatibility
# ---------------------------------------------------------------------------


def _build_tracking_dicts(params: Dict, episodes: int, n_actions: int) -> Tuple:
    cluster_actions_dict = {str(ep): {str(ac): 0 for ac in range(n_actions)} for ep in range(1, episodes + 1)}
    cluster_action_dict = {
        str(ep): {str(ag): {str(ac): 0 for ac in range(n_actions)} for ag in range(params["cluster_learners"])}
        for ep in range(1, episodes + 1)
    }
    cluster_reward_dict = {str(ep): {str(ag): 0 for ag in range(params["cluster_learners"])} for ep in range(1, episodes + 1)}
    scatter_actions_dict = {str(ep): {str(ac): 0 for ac in range(n_actions)} for ep in range(1, episodes + 1)}
    scatter_action_dict = {
        str(ep): {
            str(ag): {str(ac): 0 for ac in range(n_actions)}
            for ag in range(params["cluster_learners"], params["cluster_learners"] + params["scatter_learners"])
        }
        for ep in range(1, episodes + 1)
    }
    scatter_reward_dict = {
        str(ep): {str(ag): 0 for ag in range(params["cluster_learners"], params["cluster_learners"] + params["scatter_learners"])}
        for ep in range(1, episodes + 1)
    }
    cluster_dict = {str(ep): 0.0 for ep in range(1, episodes + 1)}
    return (
        cluster_dict,
        cluster_actions_dict,
        cluster_action_dict,
        cluster_reward_dict,
        scatter_actions_dict,
        scatter_action_dict,
        scatter_reward_dict,
    )


def _update_metrics(
    env,
    ep: int,
    action: int,
    reward: float,
    agent_id: int,
    cluster_actions_dict: Dict,
    cluster_action_dict: Dict,
    cluster_reward_dict: Dict,
    scatter_actions_dict: Dict,
    scatter_action_dict: Dict,
    scatter_reward_dict: Dict,
) -> None:
    if env.learners[agent_id]["mode"] == "c":
        cluster_actions_dict[str(ep)][str(action)] += 1
        cluster_action_dict[str(ep)][str(agent_id)][str(action)] += 1
        cluster_reward_dict[str(ep)][str(agent_id)] += round(reward, 2)
    elif env.learners[agent_id]["mode"] == "s":
        scatter_actions_dict[str(ep)][str(action)] += 1
        scatter_action_dict[str(ep)][str(agent_id)][str(action)] += 1
        scatter_reward_dict[str(ep)][str(agent_id)] += round(reward, 2)


def _append_cluster_metrics(
    env,
    ep: int,
    params: Dict,
    tick: int,
    cluster_dict: Dict,
    only_cluster_dict: Dict,
    mixed_cluster_dict: Dict,
    only_scatter_dict: Dict,
    mixed_scatter_dict: Dict,
) -> Tuple:
    del params, tick
    if env.cluster_learners == 0 or env.scatter_learners == 0:
        avg_cluster = env.avg_cluster()
        cluster_dict[str(ep)] += round(avg_cluster, 2)
        return (avg_cluster, None, None, None, None)

    avg_only_cluster, avg_mixed_cluster, avg_only_scatter, avg_mixed_scatter = env.avg_cluster()
    only_cluster_dict[str(ep)] += round(avg_only_cluster, 2)
    mixed_cluster_dict[str(ep)] += round(avg_mixed_cluster, 2)
    only_scatter_dict[str(ep)] += round(avg_only_scatter, 2)
    mixed_scatter_dict[str(ep)] += round(avg_mixed_scatter, 2)
    return (None, avg_only_cluster, avg_mixed_cluster, avg_only_scatter, avg_mixed_scatter)


def _log_episode_metrics(
    env,
    params: Dict,
    ep: int,
    tick: int,
    train_log_every: int,
    print_metrics: int,
    logger,
    cluster_dict: Dict,
    cluster_actions_dict: Dict,
    cluster_action_dict: Dict,
    cluster_reward_dict: Dict,
    scatter_actions_dict: Dict,
    scatter_action_dict: Dict,
    scatter_reward_dict: Dict,
    only_cluster_dict: Dict,
    mixed_cluster_dict: Dict,
    only_scatter_dict: Dict,
    mixed_scatter_dict: Dict,
    label: str,
) -> None:
    if ep % train_log_every != 0:
        return

    value = [ep, tick * ep]
    if env.cluster_learners == 0 or env.scatter_learners == 0:
        avg_cluster = round(cluster_dict[str(ep)] / params["episode_ticks"], 2)
        value.append(avg_cluster)
    else:
        avg_only_cluster = round(only_cluster_dict[str(ep)] / params["episode_ticks"], 2)
        avg_mixed_cluster = round(mixed_cluster_dict[str(ep)] / params["episode_ticks"], 2)
        avg_only_scatter = round(only_scatter_dict[str(ep)] / params["episode_ticks"], 2)
        avg_mixed_scatter = round(mixed_scatter_dict[str(ep)] / params["episode_ticks"], 2)
        value.extend([avg_only_cluster, avg_mixed_cluster, avg_only_scatter, avg_mixed_scatter])

    cluster_avg_rew = None
    scatter_avg_rew = None
    if params["cluster_learners"] > 0:
        cluster_avg_rew = round((sum(cluster_reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["cluster_learners"], 4)
        value.append(cluster_avg_rew)
        value.extend(list(cluster_actions_dict[str(ep)].values()))
        tmp = [list(v.values()) for v in cluster_action_dict[str(ep)].values()]
        value.extend(list(itertools.chain(*tmp)))

    if params["scatter_learners"] > 0:
        scatter_avg_rew = round((sum(scatter_reward_dict[str(ep)].values()) / params["episode_ticks"]) / params["scatter_learners"], 4)
        value.append(scatter_avg_rew)
        value.extend(list(scatter_actions_dict[str(ep)].values()))
        tmp = [list(v.values()) for v in scatter_action_dict[str(ep)].values()]
        value.extend(list(itertools.chain(*tmp)))

    value = _align_logger_row_with_schema(value, logger, label)
    logger.load_value(value)

    if print_metrics > 0 and ep % print_metrics == 0:
        print(f"\nMetrics ({label})")
        if env.cluster_learners > 0 and env.scatter_learners == 0:
            print(" - cluster: ", avg_cluster)
            print(" - cluster_reward: ", cluster_avg_rew)
        elif env.cluster_learners == 0 and env.scatter_learners > 0:
            print(" - cluster: ", avg_cluster)
            print(" - scatter_reward: ", scatter_avg_rew)
        else:
            print(" - only_cluster: ", avg_only_cluster)
            print(" - mixed_cluster: ", avg_mixed_cluster)
            print(" - cluster_reward: ", cluster_avg_rew)
            print("\n")
            print(" - only_scatter: ", avg_only_scatter)
            print(" - mixed_scatter: ", avg_mixed_scatter)
            print(" - scatter_reward: ", scatter_avg_rew)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + float(gamma) * nonterminal * next_values[t] - values[t]
        gae = delta + float(gamma) * float(gae_lambda) * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages.detach(), returns.detach()


def _ppo_update_mode(
    buffer: ModeRolloutBuffer,
    agent_ids: List[int],
    obs_dim: int,
    actor_nets: List[ActorPolicy],
    critic: CentralValueCritic,
    gamma: float,
    gae_lambda: float,
    clip_ratio: float,
    ppo_epochs: int,
    minibatch_size: int,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    normalize_advantages: bool,
) -> Dict[str, float]:
    """Update one same-mode MAPPO team only."""

    transitions = buffer.as_list()
    if not transitions:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    device = critic.device
    n_mode_agents = len(agent_ids)
    states = np.stack([tr.global_state for tr in transitions]).astype(np.float32)
    next_states = np.stack([tr.next_global_state for tr in transitions]).astype(np.float32)
    actions = np.stack([tr.actions for tr in transitions]).astype(np.int64)
    old_log_probs = np.stack([tr.old_log_probs for tr in transitions]).astype(np.float32)
    rewards = np.asarray([tr.team_reward for tr in transitions], dtype=np.float32)
    dones = np.asarray([tr.done for tr in transitions], dtype=np.float32)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
    old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)

    with torch.no_grad():
        values_t = critic(states_t)
        next_values_t = critic(next_states_t)
        advantages_t, returns_t = _compute_gae(rewards_t, values_t, next_values_t, dones_t, gamma, gae_lambda)
        if normalize_advantages and advantages_t.numel() > 1:
            std = advantages_t.std(unbiased=False)
            if float(std.item()) > 1e-8:
                advantages_t = (advantages_t - advantages_t.mean()) / (std + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0
    n_samples = states_t.shape[0]
    minibatch_size = max(1, int(minibatch_size))
    ppo_epochs = max(1, int(ppo_epochs))

    unique_actors = _unique_modules(actor_nets[agent_id] for agent_id in agent_ids)

    for _ in range(ppo_epochs):
        permutation = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            mb_states = states_t[idx]
            mb_actions = actions_t[idx]
            mb_old_log_probs = old_log_probs_t[idx]
            mb_advantages = advantages_t[idx]
            mb_returns = returns_t[idx]

            actor_loss_terms = []
            entropy_terms = []
            for local_i, agent_id in enumerate(agent_ids):
                obs_i = mb_states[:, local_i * obs_dim : (local_i + 1) * obs_dim]
                dist = actor_nets[agent_id].distribution(obs_i)
                new_log_prob = dist.log_prob(mb_actions[:, local_i])
                ratio = torch.exp(new_log_prob - mb_old_log_probs[:, local_i])
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_advantages
                actor_loss_terms.append(-torch.min(unclipped, clipped).mean())
                entropy_terms.append(dist.entropy().mean())

            policy_loss = torch.stack(actor_loss_terms).mean()
            entropy = torch.stack(entropy_terms).mean()
            values_pred = critic(mb_states)
            value_loss = F.mse_loss(values_pred, mb_returns)
            loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy

            for actor in unique_actors:
                actor.optimizer.zero_grad()
            critic.optimizer.zero_grad()
            loss.backward()

            if max_grad_norm and float(max_grad_norm) > 0:
                for actor in unique_actors:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), float(max_grad_norm))
                torch.nn.utils.clip_grad_norm_(critic.parameters(), float(max_grad_norm))

            for actor in unique_actors:
                actor.optimizer.step()
            critic.optimizer.step()

            total_policy_loss += float(policy_loss.detach().cpu().item())
            total_value_loss += float(value_loss.detach().cpu().item())
            total_entropy += float(entropy.detach().cpu().item())
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(1, n_updates),
        "value_loss": total_value_loss / max(1, n_updates),
        "entropy": total_entropy / max(1, n_updates),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_agent(params: Dict, l_params: Dict, n_obs: int, n_actions: int, train: bool) -> Tuple:
    n_agents = int(params["cluster_learners"] + params["scatter_learners"])
    episodes = int(l_params["train_episodes"] if train else l_params["test_episodes"])

    device = _resolve_device(l_params.get("device", "auto"))
    print(f"[MAPPO] using device: {device}")

    actor_hidden_dim = int(l_params.get("actor_hidden_dim", 64))
    critic_hidden_dim = int(l_params.get("critic_hidden_dim", 64))
    actor_lr = float(l_params.get("actor_learning_rate", l_params.get("alpha", 3e-4)))
    critic_lr = float(l_params.get("critic_learning_rate", actor_lr))
    gamma = float(l_params.get("gamma", 0.99))
    parameter_sharing = str(l_params.get("actor_parameter_sharing", "none"))

    agent_ids_by_mode = _agent_ids_by_mode_from_counts(params["cluster_learners"], params["scatter_learners"])
    actor_nets = _create_actor_nets(
        agent_ids_by_mode=agent_ids_by_mode,
        n_agents=n_agents,
        obs_dim=n_obs,
        hidden_dim=actor_hidden_dim,
        n_actions=n_actions,
        learning_rate=actor_lr,
        device=device,
        parameter_sharing=parameter_sharing,
    )
    critics = _create_mode_critics(agent_ids_by_mode, n_obs, critic_hidden_dim, critic_lr, device)

    (
        cluster_dict,
        cluster_actions_dict,
        cluster_action_dict,
        cluster_reward_dict,
        scatter_actions_dict,
        scatter_action_dict,
        scatter_reward_dict,
    ) = _build_tracking_dicts(params, episodes, n_actions)

    if train:
        return (
            actor_nets,
            critics,
            device,
            actor_lr,
            critic_lr,
            gamma,
            episodes,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
        )

    return (
        actor_nets,
        critics,
        device,
        episodes,
        cluster_dict,
        cluster_actions_dict,
        cluster_action_dict,
        cluster_reward_dict,
        scatter_actions_dict,
        scatter_action_dict,
        scatter_reward_dict,
    )


def train(
    env,
    params: Dict,
    l_params: Dict,
    actor_nets: List[ActorPolicy],
    critics: Dict[str, CentralValueCritic],
    cluster_dict: Dict,
    cluster_actions_dict: Dict,
    cluster_action_dict: Dict,
    cluster_reward_dict: Dict,
    scatter_actions_dict: Dict,
    scatter_action_dict: Dict,
    scatter_reward_dict: Dict,
    train_episodes: int,
    train_log_every: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
    gamma: float,
    print_metrics: int,
    logger,
    visualizer=None,
) -> Tuple[List[ActorPolicy], Dict[str, CentralValueCritic]]:
    del actor_learning_rate, critic_learning_rate  # optimizers are bound to modules

    n_obs = int(env.observations_n())
    n_agents = int(env.cluster_learners + env.scatter_learners)
    agent_ids_by_mode = _agent_ids_by_mode_from_env(env)
    _assert_mode_critics(critics, agent_ids_by_mode, n_obs)

    rollout_steps = int(l_params.get("rollout_steps", l_params.get("update_after_steps", 256)))
    ppo_epochs = int(l_params.get("ppo_epochs", 4))
    minibatch_size = int(l_params.get("minibatch_size", l_params.get("batch_size", 64)))
    gae_lambda = float(l_params.get("gae_lambda", 0.95))
    clip_ratio = float(l_params.get("clip_ratio", 0.2))
    value_coef = float(l_params.get("value_coef", 0.5))
    entropy_coef = float(l_params.get("entropy_coef", 0.01))
    max_grad_norm = float(l_params.get("max_grad_norm", 0.5))
    normalize_advantages = bool(l_params.get("normalize_advantages", True))

    buffers = {mode: ModeRolloutBuffer() for mode in agent_ids_by_mode}
    prev_snapshot: Dict[int, Dict] = {}
    global_mode_steps = {mode: 0 for mode in agent_ids_by_mode}

    only_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    only_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}

    print("Start training (MAPPO, same-mode centralized critics)...\n")

    for ep in tqdm(range(1, int(train_episodes) + 1), desc="EPISODES", colour="red", position=0, leave=False):
        env.reset()
        prev_snapshot = {}

        for tick in tqdm(range(1, int(params["episode_ticks"]) + 1), desc="TICKS", colour="green", position=1, leave=False):
            snapshot: Dict[int, Dict] = {}

            for agent in env.agent_iter(max_iter=n_agents):
                cur_state, reward, _, _, _ = env.last(agent)
                agent_id = int(agent)
                obs_vec = _as_obs_vec(cur_state, n_obs)
                action, log_prob, entropy = actor_nets[agent_id].act(obs_vec)
                env.step(action)

                snapshot[agent_id] = {
                    "obs_vec": obs_vec,
                    "action": int(action),
                    "log_prob": float(log_prob),
                    "entropy": float(entropy),
                    "reward": float(reward),
                }

                _update_metrics(
                    env,
                    ep,
                    int(action),
                    float(reward),
                    agent_id,
                    cluster_actions_dict,
                    cluster_action_dict,
                    cluster_reward_dict,
                    scatter_actions_dict,
                    scatter_action_dict,
                    scatter_reward_dict,
                )

            if prev_snapshot:
                done = 1.0 if tick == int(params["episode_ticks"]) else 0.0
                for mode, agent_ids in agent_ids_by_mode.items():
                    if not all(ag in prev_snapshot and ag in snapshot for ag in agent_ids):
                        continue
                    global_state = np.concatenate([prev_snapshot[ag]["obs_vec"] for ag in agent_ids]).astype(np.float32)
                    next_global_state = np.concatenate([snapshot[ag]["obs_vec"] for ag in agent_ids]).astype(np.float32)
                    actions = np.asarray([prev_snapshot[ag]["action"] for ag in agent_ids], dtype=np.int64)
                    old_log_probs = np.asarray([prev_snapshot[ag]["log_prob"] for ag in agent_ids], dtype=np.float32)
                    team_reward = float(sum(prev_snapshot[ag]["reward"] for ag in agent_ids))
                    buffers[mode].push(
                        ModeTransition(
                            global_state=global_state,
                            actions=actions,
                            old_log_probs=old_log_probs,
                            team_reward=team_reward,
                            next_global_state=next_global_state,
                            done=done,
                        )
                    )
                    global_mode_steps[mode] += 1

                    if len(buffers[mode]) >= max(1, rollout_steps):
                        _ppo_update_mode(
                            buffer=buffers[mode],
                            agent_ids=agent_ids,
                            obs_dim=n_obs,
                            actor_nets=actor_nets,
                            critic=critics[mode],
                            gamma=gamma,
                            gae_lambda=gae_lambda,
                            clip_ratio=clip_ratio,
                            ppo_epochs=ppo_epochs,
                            minibatch_size=minibatch_size,
                            value_coef=value_coef,
                            entropy_coef=entropy_coef,
                            max_grad_norm=max_grad_norm,
                            normalize_advantages=normalize_advantages,
                        )
                        buffers[mode].clear()

            prev_snapshot = snapshot

            _append_cluster_metrics(
                env,
                ep,
                params,
                tick,
                cluster_dict,
                only_cluster_dict,
                mixed_cluster_dict,
                only_scatter_dict,
                mixed_scatter_dict,
            )

            if visualizer is not None:
                visualizer.render(env.patches, env.learners, env.fov, env.ph_fov)

        # Flush any partial same-mode rollouts at the end of each episode.
        for mode, agent_ids in agent_ids_by_mode.items():
            if len(buffers[mode]) > 0:
                _ppo_update_mode(
                    buffer=buffers[mode],
                    agent_ids=agent_ids,
                    obs_dim=n_obs,
                    actor_nets=actor_nets,
                    critic=critics[mode],
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_ratio=clip_ratio,
                    ppo_epochs=ppo_epochs,
                    minibatch_size=minibatch_size,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    max_grad_norm=max_grad_norm,
                    normalize_advantages=normalize_advantages,
                )
                buffers[mode].clear()

        _log_episode_metrics(
            env,
            params,
            ep,
            tick,
            train_log_every,
            print_metrics,
            logger,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
            only_cluster_dict,
            mixed_cluster_dict,
            only_scatter_dict,
            mixed_scatter_dict,
            "MAPPO",
        )

    logger.empty_table()
    env.close()
    if visualizer is not None:
        visualizer.close()
    print("Training finished (MAPPO)!\n")

    return actor_nets, critics


def eval(
    env,
    params: Dict,
    cluster_dict: Dict,
    cluster_actions_dict: Dict,
    cluster_action_dict: Dict,
    cluster_reward_dict: Dict,
    scatter_actions_dict: Dict,
    scatter_action_dict: Dict,
    scatter_reward_dict: Dict,
    test_episodes: int,
    actor_nets: List[ActorPolicy],
    test_log_every: int,
    logger,
    visualizer=None,
    l_params: Optional[Dict] = None,
    evaluation_policy: Optional[str] = None,
) -> None:
    n_obs = int(env.observations_n())
    n_agents = int(env.cluster_learners + env.scatter_learners)
    evaluation_policy = resolve_evaluation_policy(l_params, evaluation_policy=evaluation_policy)

    only_cluster_dict = {str(ep): 0.0 for ep in range(1, int(test_episodes) + 1)}
    mixed_cluster_dict = {str(ep): 0.0 for ep in range(1, int(test_episodes) + 1)}
    only_scatter_dict = {str(ep): 0.0 for ep in range(1, int(test_episodes) + 1)}
    mixed_scatter_dict = {str(ep): 0.0 for ep in range(1, int(test_episodes) + 1)}

    print(f"Start evaluation (MAPPO, {evaluation_policy} policy)...\n")

    for ep in tqdm(range(1, int(test_episodes) + 1), desc="EPISODES", colour="red", position=0, leave=False):
        env.reset()

        for tick in tqdm(range(1, int(params["episode_ticks"]) + 1), desc="TICKS", colour="green", position=1, leave=False):
            for agent in env.agent_iter(max_iter=n_agents):
                cur_state, reward, _, _, _ = env.last(agent)
                agent_id = int(agent)
                obs_vec = _as_obs_vec(cur_state, n_obs)
                action = _select_evaluation_action(actor_nets[agent_id], obs_vec, evaluation_policy)
                env.step(action)

                _update_metrics(
                    env,
                    ep,
                    int(action),
                    float(reward),
                    agent_id,
                    cluster_actions_dict,
                    cluster_action_dict,
                    cluster_reward_dict,
                    scatter_actions_dict,
                    scatter_action_dict,
                    scatter_reward_dict,
                )

            _append_cluster_metrics(
                env,
                ep,
                params,
                tick,
                cluster_dict,
                only_cluster_dict,
                mixed_cluster_dict,
                only_scatter_dict,
                mixed_scatter_dict,
            )

            if visualizer is not None:
                visualizer.render(env.patches, env.learners, env.fov, env.ph_fov)

        _log_episode_metrics(
            env,
            params,
            ep,
            tick,
            test_log_every,
            0,
            logger,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
            only_cluster_dict,
            mixed_cluster_dict,
            only_scatter_dict,
            mixed_scatter_dict,
            "MAPPO eval",
        )

    logger.empty_table()
    env.close()
    if visualizer is not None:
        visualizer.close()
    print("Evaluation finished (MAPPO)!\n")
