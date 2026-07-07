"""
QMIX with Neural Networks — Alternative C (Replay Buffer + Mini-batch + Targets).

This module keeps the existing runner-facing API but changes training to an
off-policy deep-RL workflow:

1) Collect joint transitions into a replay buffer
2) Sample mini-batches
3) Compute QMIX TD loss with target networks
4) Update online networks
5) Periodically sync targets (hard or Polyak)
"""

import itertools
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _one_hot(index: int, size: int) -> NDArray:
    v = np.zeros(size, dtype=np.float32)
    v[int(index)] = 1.0
    return v


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    global_state: NDArray
    actions: NDArray
    team_reward: float
    next_global_state: NDArray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._data: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._data)

    def push(self, transition: Transition) -> None:
        self._data.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(list(self._data), int(batch_size))


# ---------------------------------------------------------------------------
# Neural modules
# ---------------------------------------------------------------------------


class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int, learning_rate: float, device: torch.device) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, state_vec: NDArray) -> NDArray:
        with torch.no_grad():
            t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device)
            return self.forward(t).cpu().numpy()

    def update_td(self, state_vec: NDArray, action: int, target: float) -> None:
        t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device)
        q_values = self.forward(t)
        q_target = q_values.detach().clone()
        q_target[int(action)] = float(target)
        loss = F.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def max_q(self, state_vec: NDArray) -> float:
        return float(np.max(self.predict(state_vec)))


class QMIXMixer(nn.Module):
    """
    Monotonic mixer. Supports both single sample ([state_dim]) and batch
    ([B, state_dim]) inputs.
    """

    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int, learning_rate: float, device: torch.device) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.device = device

        self.hyper_w = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
        )
        self.hyper_b = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, global_state: torch.Tensor, agent_qs: torch.Tensor) -> torch.Tensor:
        # Works for shapes:
        # global_state [state_dim], agent_qs [n_agents] -> scalar
        # global_state [B, state_dim], agent_qs [B, n_agents] -> [B]
        weights = F.softplus(self.hyper_w(global_state))
        bias = self.hyper_b(global_state).squeeze(-1)
        return (weights * agent_qs).sum(dim=-1) + bias

    def predict(self, global_state_vec: NDArray, agent_qs: NDArray) -> float:
        with torch.no_grad():
            gs = torch.as_tensor(global_state_vec, dtype=torch.float32, device=self.device)
            aqs = torch.as_tensor(agent_qs, dtype=torch.float32, device=self.device)
            return float(self.forward(gs, aqs).item())

    def update_td(self, prev_global_state: NDArray, prev_agent_qs: NDArray, team_reward: float, q_total_next: float, gamma: float) -> None:
        gs = torch.as_tensor(prev_global_state, dtype=torch.float32, device=self.device)
        aqs = torch.as_tensor(prev_agent_qs, dtype=torch.float32, device=self.device)
        q_total = self.forward(gs, aqs)
        target = torch.tensor(team_reward + gamma * q_total_next, dtype=torch.float32, device=self.device)
        loss = F.mse_loss(q_total, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_model(agent_nets: List[AgentQNetwork], mixer: QMIXMixer, path: str) -> None:
    torch.save(
        {
            "agent_nets": [net.state_dict() for net in agent_nets],
            "mixer": mixer.state_dict(),
        },
        path,
    )


def load_model(agent_nets: List[AgentQNetwork], mixer: QMIXMixer, path: str, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    for i, state_dict in enumerate(checkpoint["agent_nets"]):
        agent_nets[i].load_state_dict(state_dict)
    mixer.load_state_dict(checkpoint["mixer"])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_agent(params: Dict, l_params: Dict, n_obs: int, n_actions: int, train: bool) -> Tuple:
    n_agents = params["cluster_learners"] + params["scatter_learners"]
    episodes = l_params["train_episodes"] if train else l_params["test_episodes"]

    device = _resolve_device(l_params.get("device", "auto"))
    print(f"[QMIX-NN] using device: {device}")

    hidden_dim = int(l_params.get("agent_hidden_dim", 32))
    mixer_hidden_dim = int(l_params.get("mixer_hidden_dim", 32))
    agent_lr = float(l_params.get("alpha", l_params.get("agent_learning_rate", 0.001)))
    mixer_lr = float(l_params.get("mixer_learning_rate", agent_lr))
    gamma = float(l_params.get("gamma", 0.9))
    epsilon = float(l_params.get("epsilon", 1.0))
    epsilon_min = float(l_params.get("epsilon_min", 0.1))
    decay_type = l_params.get("decay_type", "log")
    decay = float(l_params.get("decay", 0.9987))

    agent_nets = [AgentQNetwork(n_obs, hidden_dim, n_actions, agent_lr, device) for _ in range(n_agents)]
    mixer = QMIXMixer(n_agents=n_agents, state_dim=n_agents * n_obs, hidden_dim=mixer_hidden_dim, learning_rate=mixer_lr, device=device)

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
        )

    return (
        agent_nets,
        mixer,
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


def _sample_and_update(
    replay: ReplayBuffer,
    batch_size: int,
    n_agents: int,
    n_obs: int,
    n_actions: int,
    gamma: float,
    device: torch.device,
    agent_nets: List[AgentQNetwork],
    mixer: QMIXMixer,
    target_agent_nets: List[AgentQNetwork],
    target_mixer: QMIXMixer,
) -> None:
    batch = replay.sample(batch_size)

    states = np.stack([tr.global_state for tr in batch]).astype(np.float32)          # [B, state_dim]
    actions = np.stack([tr.actions for tr in batch]).astype(np.int64)                # [B, n_agents]
    rewards = np.asarray([tr.team_reward for tr in batch], dtype=np.float32)         # [B]
    next_states = np.stack([tr.next_global_state for tr in batch]).astype(np.float32)
    dones = np.asarray([tr.done for tr in batch], dtype=np.float32)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)

    # Split global state into per-agent observation slices
    # obs slice for agent i is [i*n_obs : (i+1)*n_obs]
    online_selected_qs = []
    target_next_max_qs = []

    for i in range(n_agents):
        obs_i = states_t[:, i * n_obs : (i + 1) * n_obs]                 # [B, n_obs]
        next_obs_i = next_states_t[:, i * n_obs : (i + 1) * n_obs]       # [B, n_obs]

        q_i = agent_nets[i](obs_i)                                        # [B, n_actions]
        q_i_selected = q_i.gather(1, actions_t[:, i].unsqueeze(1)).squeeze(1)  # [B]
        online_selected_qs.append(q_i_selected)

        with torch.no_grad():
            q_next_i = target_agent_nets[i](next_obs_i)                   # [B, n_actions]
            q_next_i_max = q_next_i.max(dim=1).values                     # [B]
            target_next_max_qs.append(q_next_i_max)

    online_selected_qs_t = torch.stack(online_selected_qs, dim=1)         # [B, n_agents]
    target_next_max_qs_t = torch.stack(target_next_max_qs, dim=1)         # [B, n_agents]

    q_tot = mixer(states_t, online_selected_qs_t)                         # [B]
    with torch.no_grad():
        q_tot_next = target_mixer(next_states_t, target_next_max_qs_t)    # [B]
        td_target = rewards_t + gamma * (1.0 - dones_t) * q_tot_next      # [B]

    loss = F.mse_loss(q_tot, td_target)

    # Joint optimization over all online nets + mixer
    for net in agent_nets:
        net.optimizer.zero_grad()
    mixer.optimizer.zero_grad()

    loss.backward()

    for net in agent_nets:
        net.optimizer.step()
    mixer.optimizer.step()


def train(
    env,
    params: Dict,
    l_params: Dict,
    agent_nets: List[AgentQNetwork],
    mixer: QMIXMixer,
    cluster_dict: Dict,
    cluster_actions_dict: Dict,
    cluster_action_dict: Dict,
    cluster_reward_dict: Dict,
    scatter_actions_dict: Dict,
    scatter_action_dict: Dict,
    scatter_reward_dict: Dict,
    train_episodes: int,
    train_log_every: int,
    agent_learning_rate: float,
    mixer_learning_rate: float,
    gamma: float,
    decay_type: str,
    decay: float,
    epsilon: float,
    epsilon_min: float,
    print_metrics: int,
    logger,
    visualizer=None,
) -> Tuple[List[AgentQNetwork], QMIXMixer]:
    del agent_learning_rate, mixer_learning_rate  # optimizers are bound to modules

    n_obs = env.observations_n()
    n_actions = env.actions_n()
    n_agents = env.cluster_learners + env.scatter_learners
    device = agent_nets[0].device if agent_nets else torch.device("cpu")

    # Alternative C hyperparameters
    replay_capacity = int(l_params.get("replay_capacity", 10000))
    batch_size = int(l_params.get("batch_size", 64))
    learning_starts = int(l_params.get("learning_starts", 256))
    train_every = int(l_params.get("train_every", 1))
    target_update_interval = int(l_params.get("target_update_interval", 200))
    target_update_mode = str(l_params.get("target_update_mode", "hard"))  # hard | soft
    tau = float(l_params.get("tau", 0.01))

    replay = ReplayBuffer(replay_capacity)

    # Target networks
    target_agent_nets = [
        AgentQNetwork(n_obs, int(l_params.get("agent_hidden_dim", 32)), n_actions, float(l_params.get("alpha", 0.001)), device)
        for _ in range(n_agents)
    ]
    for i in range(n_agents):
        _hard_update(target_agent_nets[i], agent_nets[i])

    target_mixer = QMIXMixer(
        n_agents=n_agents,
        state_dim=n_agents * n_obs,
        hidden_dim=int(l_params.get("mixer_hidden_dim", 32)),
        learning_rate=float(l_params.get("mixer_learning_rate", l_params.get("alpha", 0.001))),
        device=device,
    )
    _hard_update(target_mixer, mixer)

    old_state_vec: Dict[str, NDArray] = {}
    old_action: Dict[str, int] = {}
    prev_snapshot: Dict[int, Dict] = {}
    global_step = 0
    gradient_steps = 0

    only_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    only_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}

    print("Start training (QMIX-NN, replay + mini-batch)...\n")

    for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour="red", position=0, leave=False):
        env.reset()
        prev_snapshot = {}

        for tick in tqdm(range(1, params["episode_ticks"] + 1), desc="TICKS", colour="green", position=1, leave=False):
            snapshot: Dict[int, Dict] = {}

            for agent in env.agent_iter(max_iter=n_agents):
                cur_state, reward, _, _, _ = env.last(agent)
                agent_id = int(agent)
                state_idx = env.convert_observation(cur_state)
                state_vec = _one_hot(state_idx, n_obs)
                q_values = agent_nets[agent_id].predict(state_vec)

                if ep == 1 and tick == 1 and agent not in old_state_vec:
                    action = int(np.random.randint(0, n_actions))
                else:
                    if random.uniform(0.0, 1.0) < epsilon:
                        action = int(np.random.randint(0, n_actions))
                    else:
                        action = int(np.argmax(q_values))

                env.step(action)
                old_state_vec[agent] = state_vec
                old_action[agent] = action

                snapshot[agent_id] = {
                    "state_vec": state_vec,
                    "action": action,
                    "reward": float(reward),
                }

                _update_metrics(
                    env,
                    ep,
                    action,
                    float(reward),
                    agent_id,
                    cluster_actions_dict,
                    cluster_action_dict,
                    cluster_reward_dict,
                    scatter_actions_dict,
                    scatter_action_dict,
                    scatter_reward_dict,
                )

            # Push joint transition after we have both previous and current snapshots
            if prev_snapshot:
                ids = sorted(prev_snapshot.keys())
                global_state = np.concatenate([prev_snapshot[ag]["state_vec"] for ag in ids]).astype(np.float32)
                next_global_state = np.concatenate([snapshot[ag]["state_vec"] for ag in ids]).astype(np.float32)
                actions = np.asarray([prev_snapshot[ag]["action"] for ag in ids], dtype=np.int64)
                team_reward = float(sum(prev_snapshot[ag]["reward"] for ag in ids))
                # The current env API does not expose per-tick terminal flags in this loop.
                done = 0.0
                replay.push(
                    Transition(
                        global_state=global_state,
                        actions=actions,
                        team_reward=team_reward,
                        next_global_state=next_global_state,
                        done=done,
                    )
                )

                # Train from replay
                if len(replay) >= max(batch_size, learning_starts) and global_step % max(1, train_every) == 0:
                    _sample_and_update(
                        replay=replay,
                        batch_size=batch_size,
                        n_agents=n_agents,
                        n_obs=n_obs,
                        n_actions=n_actions,
                        gamma=gamma,
                        device=device,
                        agent_nets=agent_nets,
                        mixer=mixer,
                        target_agent_nets=target_agent_nets,
                        target_mixer=target_mixer,
                    )
                    gradient_steps += 1

                    # Target update
                    if target_update_mode == "soft":
                        for i in range(n_agents):
                            _soft_update(target_agent_nets[i], agent_nets[i], tau)
                        _soft_update(target_mixer, mixer, tau)
                    else:
                        if gradient_steps % max(1, target_update_interval) == 0:
                            for i in range(n_agents):
                                _hard_update(target_agent_nets[i], agent_nets[i])
                            _hard_update(target_mixer, mixer)

                global_step += 1

            prev_snapshot = snapshot

            if env.cluster_learners == 0 or env.scatter_learners == 0:
                cluster_dict[str(ep)] += round(env.avg_cluster(), 2)
            else:
                (
                    avg_only_cluster,
                    avg_mixed_cluster,
                    avg_only_scatter,
                    avg_mixed_scatter,
                ) = env.avg_cluster()
                only_cluster_dict[str(ep)] += round(avg_only_cluster, 2)
                mixed_cluster_dict[str(ep)] += round(avg_mixed_cluster, 2)
                only_scatter_dict[str(ep)] += round(avg_only_scatter, 2)
                mixed_scatter_dict[str(ep)] += round(avg_mixed_scatter, 2)

            if visualizer is not None:
                visualizer.render(env.patches, env.learners, env.fov, env.ph_fov)

        if decay_type == "log":
            epsilon = max(epsilon * decay, epsilon_min)
        elif decay_type == "linear":
            epsilon = max(epsilon - (1 - decay), epsilon_min)

        if ep % train_log_every == 0:
            value = [ep, tick * ep]
            if env.cluster_learners == 0 or env.scatter_learners == 0:
                avg_cluster = round(cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_cluster)
            else:
                avg_only_cluster = round(only_cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_only_cluster)
                avg_mixed_cluster = round(mixed_cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_mixed_cluster)
                avg_only_scatter = round(only_scatter_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_only_scatter)
                avg_mixed_scatter = round(mixed_scatter_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_mixed_scatter)

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

            value.append(round(epsilon, 4))
            logger.load_value(value)

            if ep % print_metrics == 0:
                print("\nMetrics (QMIX-NN)")
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
                print(" - epsilon: ", round(epsilon, 4))

    logger.empty_table()
    env.close()
    if visualizer is not None:
        visualizer.close()
    print("Training finished (QMIX-NN)!\n")

    return agent_nets, mixer


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
    agent_nets: List[AgentQNetwork],
    test_log_every: int,
    logger,
    visualizer=None,
) -> None:
    n_obs = env.observations_n()
    agents_num = env.cluster_learners + env.scatter_learners
    only_cluster_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    mixed_cluster_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    only_scatter_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    mixed_scatter_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}

    print("Start evaluation (QMIX-NN)...\n")

    for ep in tqdm(range(1, test_episodes + 1), desc="EPISODES", colour="red", position=0, leave=False):
        env.reset()

        for tick in tqdm(range(1, params["episode_ticks"] + 1), desc="TICKS", colour="green", position=1, leave=False):
            for agent in env.agent_iter(max_iter=agents_num):
                cur_state, reward, _, _, _ = env.last(agent)
                agent_id = int(agent)
                state_idx = env.convert_observation(cur_state)
                state_vec = _one_hot(state_idx, n_obs)
                q_values = agent_nets[agent_id].predict(state_vec)
                action = int(np.argmax(q_values))
                env.step(action)

                _update_metrics(
                    env,
                    ep,
                    action,
                    float(reward),
                    agent_id,
                    cluster_actions_dict,
                    cluster_action_dict,
                    cluster_reward_dict,
                    scatter_actions_dict,
                    scatter_action_dict,
                    scatter_reward_dict,
                )

            if env.cluster_learners == 0 or env.scatter_learners == 0:
                cluster_dict[str(ep)] += round(env.avg_cluster(), 2)
            else:
                (
                    avg_only_cluster,
                    avg_mixed_cluster,
                    avg_only_scatter,
                    avg_mixed_scatter,
                ) = env.avg_cluster()
                only_cluster_dict[str(ep)] += round(avg_only_cluster, 2)
                mixed_cluster_dict[str(ep)] += round(avg_mixed_cluster, 2)
                only_scatter_dict[str(ep)] += round(avg_only_scatter, 2)
                mixed_scatter_dict[str(ep)] += round(avg_mixed_scatter, 2)

            if visualizer is not None:
                visualizer.render(env.patches, env.learners, env.fov, env.ph_fov)

        if ep % test_log_every == 0:
            value = [ep, tick * ep]
            if env.cluster_learners == 0 or env.scatter_learners == 0:
                avg_cluster = round(cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_cluster)
            else:
                avg_only_cluster = round(only_cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_only_cluster)
                avg_mixed_cluster = round(mixed_cluster_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_mixed_cluster)
                avg_only_scatter = round(only_scatter_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_only_scatter)
                avg_mixed_scatter = round(mixed_scatter_dict[str(ep)] / params["episode_ticks"], 2)
                value.append(avg_mixed_scatter)

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

            logger.load_value(value)

    logger.empty_table()
    env.close()
    if visualizer is not None:
        visualizer.close()
    print("Evaluation finished (QMIX-NN)!\n")

