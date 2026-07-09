"""
QMIX: Value Decomposition Networks for Cooperative Multi-Agent Reinforcement Learning

QMIX decomposes the global Q-function into individual agent Q-functions and a mixing network.
This allows decentralized execution (each agent acts independently) while maintaining centralized training.

Key idea:
    Q_total(state) = Q_mixing(Q_1, Q_2, ..., Q_n; state)
    where Q_i are individual agent Q-values and Q_mixing is a learned function (neural network).

For this tabular environment implementation:
    - Individual Q-tables remain for decentralized action selection
    - Mixing network learns weights to aggregate individual Q-values
    - Training minimizes TD error on the mixed Q-value
"""

import itertools
import random
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


_MODES = ("c", "s")


class MixingNetwork:
    """
    Simple linear mixing network for value decomposition.

    Q_total = W * [Q_1, Q_2, ..., Q_n] + V

    Args:
        n_agents: Number of agents
        n_actions: Number of actions per agent
        learning_rate: Learning rate for mixing network weights
    """

    def __init__(self, n_agents: int, n_actions: int, learning_rate: float = 0.025):
        if int(n_agents) <= 0:
            raise ValueError("MixingNetwork requires at least one agent")
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.learning_rate = learning_rate

        # Mixing weights: one set per action (agents x 1)
        self.weights = np.random.randn(self.n_actions, self.n_agents) * 0.01
        # Value baseline
        self.value = np.zeros(self.n_actions)

    def _q_array(self, individual_q_values: List[float]) -> NDArray:
        q_array = np.asarray(individual_q_values, dtype=float).reshape(-1)
        if q_array.shape[0] != self.n_agents:
            raise ValueError(
                f"MixingNetwork expected {self.n_agents} individual Q-values, "
                f"got {q_array.shape[0]}"
            )
        return q_array

    def _validate_action(self, action_idx: int) -> int:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self.n_actions:
            raise ValueError(f"action_idx {action_idx} is outside [0, {self.n_actions})")
        return action_idx

    def forward(self, individual_q_values: List[float], action_idx: int) -> float:
        """
        Compute mixed Q-value for a given action.

        Args:
            individual_q_values: [Q_1, Q_2, ..., Q_n] for the action
            action_idx: Action index (0 to n_actions-1)

        Returns:
            Mixed Q-value
        """
        action_idx = self._validate_action(action_idx)
        q_array = self._q_array(individual_q_values)
        mixed = np.dot(self.weights[action_idx], q_array) + self.value[action_idx]
        return float(mixed)

    def backward(
        self,
        individual_q_values: List[float],
        action_idx: int,
        td_error: float,
    ) -> None:
        """
        Update mixing network weights using TD error.

        Args:
            individual_q_values: Individual Q-values [Q_1, Q_2, ..., Q_n]
            action_idx: Action index
            td_error: Temporal difference error
        """
        action_idx = self._validate_action(action_idx)
        q_array = self._q_array(individual_q_values)

        # Gradient w.r.t. weights: td_error * q_array
        grad_w = td_error * q_array
        self.weights[action_idx] -= self.learning_rate * grad_w

        # Gradient w.r.t. value: td_error
        grad_v = td_error
        self.value[action_idx] -= self.learning_rate * grad_v


def _agent_ids_by_mode_from_params(params: Dict) -> Dict[str, List[int]]:
    """
    Return the fixed Slime learner id partition by kind/mode.

    Slime creates cluster learners first and scatter learners afterwards. Keeping
    separate mixer groups for these ids prevents QMIX centralized training from
    using the other kind's rewards, actions, or Q-values as shared learning data.
    """

    cluster_count = int(params["cluster_learners"])
    scatter_count = int(params["scatter_learners"])
    groups = {
        "c": list(range(cluster_count)),
        "s": list(range(cluster_count, cluster_count + scatter_count)),
    }
    return {mode: ids for mode, ids in groups.items() if ids}


def _agent_ids_by_mode_from_env(env, agents_num: int) -> Dict[str, List[int]]:
    """
    Return learner ids grouped by env.learners[*]["mode"].
    """

    groups = {mode: [] for mode in _MODES}
    for agent_id in range(agents_num):
        mode = env.learners[agent_id]["mode"]
        if mode not in groups:
            raise ValueError(f"Unsupported learner mode for agent {agent_id}: {mode!r}")
        groups[mode].append(agent_id)
    return {mode: ids for mode, ids in groups.items() if ids}


def _build_mixing_networks(params: Dict, n_actions: int, learning_rate: float) -> Dict[str, MixingNetwork]:
    """
    Build one independent mixer per non-empty learner kind.
    """

    return {
        mode: MixingNetwork(n_agents=len(agent_ids), n_actions=n_actions, learning_rate=learning_rate)
        for mode, agent_ids in _agent_ids_by_mode_from_params(params).items()
    }


def _ensure_mode_mixing_networks(
    mixing_net,
    agent_ids_by_mode: Dict[str, List[int]],
) -> Dict[str, MixingNetwork]:
    """
    Validate that training uses independent same-mode mixers.

    A legacy single all-agent mixer is rejected when both modes are present,
    because it is a cross-mode information channel.
    """

    if isinstance(mixing_net, MixingNetwork):
        if len(agent_ids_by_mode) == 1:
            mode, agent_ids = next(iter(agent_ids_by_mode.items()))
            if mixing_net.n_agents != len(agent_ids):
                raise ValueError(
                    f"Legacy single mixer has {mixing_net.n_agents} agents, "
                    f"but mode {mode!r} has {len(agent_ids)} agents"
                )
            return {mode: mixing_net}
        raise ValueError(
            "QMIX same-mode constraint requires a dict of independent mixers "
            "when both cluster ('c') and scatter ('s') learners are present"
        )

    if not isinstance(mixing_net, dict):
        raise TypeError("mixing_net must be a dict mapping learner mode to MixingNetwork")

    validated: Dict[str, MixingNetwork] = {}
    for mode, agent_ids in agent_ids_by_mode.items():
        if mode not in mixing_net:
            raise ValueError(f"Missing MixingNetwork for learner mode {mode!r}")
        net = mixing_net[mode]
        if not isinstance(net, MixingNetwork):
            raise TypeError(f"Mixer for learner mode {mode!r} is not a MixingNetwork")
        if net.n_agents != len(agent_ids):
            raise ValueError(
                f"Mixer for mode {mode!r} has {net.n_agents} agents, "
                f"but the environment has {len(agent_ids)} agents of that mode"
            )
        validated[mode] = net

    extra_modes = set(mixing_net) - set(agent_ids_by_mode)
    if extra_modes:
        raise ValueError(f"Mixing networks provided for inactive modes: {sorted(extra_modes)}")

    return validated


def _update_mode_mixing_networks(
    qtable: NDArray,
    mixing_nets: Dict[str, MixingNetwork],
    step_info: Dict[int, Dict],
    old_s: Dict[str, int],
    old_a: Dict[str, int],
    agent_ids_by_mode: Dict[str, List[int]],
    n_actions: int,
    gamma: float,
) -> None:
    """
    Update each mode-specific mixer using only same-mode agent data.
    """

    for mode, agent_ids in agent_ids_by_mode.items():
        mode_step_ids = [agent_id for agent_id in agent_ids if agent_id in step_info]
        if not mode_step_ids:
            continue
        if len(mode_step_ids) != len(agent_ids):
            continue

        if any(str(agent_id) not in old_s or str(agent_id) not in old_a for agent_id in mode_step_ids):
            continue

        mode_mixer = mixing_nets[mode]
        individual_q_values = [
            float(qtable[agent_id, old_s[str(agent_id)], old_a[str(agent_id)]])
            for agent_id in mode_step_ids
        ]
        next_individual_q_values = [
            float(np.max(qtable[agent_id, step_info[agent_id]["state"], :]))
            for agent_id in mode_step_ids
        ]
        reward_sum = sum(float(step_info[agent_id]["reward"]) for agent_id in mode_step_ids)

        for agent_id in mode_step_ids:
            old_a_int = int(old_a[str(agent_id)])
            mixed_q = mode_mixer.forward(individual_q_values, old_a_int)
            mixed_q_next = max(
                mode_mixer.forward(next_individual_q_values, action_idx)
                for action_idx in range(n_actions)
            )
            mixed_target = reward_sum + gamma * mixed_q_next
            td_error = mixed_target - mixed_q
            mode_mixer.backward(individual_q_values, old_a_int, td_error)


def create_agent(
    params: Dict,
    l_params: Dict,
    n_obs: int,
    n_actions: int,
    train: bool,
) -> Tuple:
    """
    Initialize Q-tables, mixing network, and logging structures for QMIX training/eval.

    Args:
        params: Environment parameters (cluster_learners, scatter_learners, etc.)
        l_params: Learning parameters (alpha, gamma, epsilon, etc.)
        n_obs: Number of observation states
        n_actions: Number of actions
        train: If True, return training structures; else evaluation structures

    Returns:
        Tuple of (qtable, mixing_net, alpha, gamma, epsilon, epsilon_min, decay_type,
                  decay, episodes, cluster_dict, cluster_actions_dict, cluster_action_dict,
                  cluster_reward_dict, scatter_actions_dict, scatter_action_dict, scatter_reward_dict)
    """

    learner_population = params["cluster_learners"] + params["scatter_learners"]
    episodes = l_params["train_episodes"] if train else l_params["test_episodes"]

    # Initialize per-agent Q-tables (same structure as IQL)
    qtable = np.zeros([learner_population, n_obs, n_actions])

    # Initialize one independent mixer per learner kind. This enforces the
    # constraint that cluster learners share centralized-training information
    # only with cluster learners, and scatter learners only with scatter learners.
    mixing_learning_rate = l_params.get("mixing_learning_rate", l_params.get("alpha", 0.025))
    mixing_net = _build_mixing_networks(params, n_actions, mixing_learning_rate)

    # Action frequency tracking per episode
    cluster_actions_dict = {
        str(ep): {str(ac): 0 for ac in range(n_actions)}
        for ep in range(1, episodes + 1)
    }

    # Action frequency per cluster agent per episode
    cluster_action_dict = {
        str(ep): {
            str(ag): {str(ac): 0 for ac in range(n_actions)}
            for ag in range(params["cluster_learners"])
        }
        for ep in range(1, episodes + 1)
    }

    # Cumulative reward per cluster agent per episode
    cluster_reward_dict = {
        str(ep): {str(ag): 0 for ag in range(params["cluster_learners"])}
        for ep in range(1, episodes + 1)
    }

    # Action frequency tracking for scatter agents
    scatter_actions_dict = {
        str(ep): {str(ac): 0 for ac in range(n_actions)}
        for ep in range(1, episodes + 1)
    }

    # Action frequency per scatter agent per episode
    scatter_action_dict = {
        str(ep): {
            str(ag): {str(ac): 0 for ac in range(n_actions)}
            for ag in range(params["cluster_learners"], learner_population)
        }
        for ep in range(1, episodes + 1)
    }

    # Cumulative reward per scatter agent per episode
    scatter_reward_dict = {
        str(ep): {str(ag): 0 for ag in range(params["cluster_learners"], learner_population)}
        for ep in range(1, episodes + 1)
    }

    # Cumulative clustering metric per episode
    cluster_dict = {str(ep): 0.0 for ep in range(1, episodes + 1)}

    if train:
        alpha = l_params["alpha"]
        gamma = l_params["gamma"]
        epsilon = l_params["epsilon"]
        epsilon_min = l_params["epsilon_min"]
        decay_type = l_params["decay_type"]
        decay = l_params["decay"]

        return (
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
        )
    else:
        return (
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
    qtable: NDArray,
    mixing_net: Dict[str, MixingNetwork],
    cluster_dict: Dict,
    cluster_actions_dict: Dict,
    cluster_action_dict: Dict,
    cluster_reward_dict: Dict,
    scatter_actions_dict: Dict,
    scatter_action_dict: Dict,
    scatter_reward_dict: Dict,
    train_episodes: int,
    train_log_every: int,
    alpha: float,
    gamma: float,
    decay_type: str,
    decay: float,
    epsilon: float,
    epsilon_min: float,
    print_metrics: int,
    logger,
    visualizer=None,
) -> Tuple[NDArray, Dict[str, MixingNetwork]]:
    """
    Train QMIX agent(s) on the environment.

    Each agent maintains its own Q-table for decentralized execution.
    The mixing network aggregates individual Q-values for centralized learning.

    Args:
        env: The environment (PettingZoo-compatible)
        params: Environment configuration
        l_params: Learning parameters
        qtable: Per-agent Q-tables [n_agents, n_obs, n_actions]
        mixing_net: Mode-specific mixing networks for value decomposition
        cluster_dict: Cumulative metrics tracking
        cluster_actions_dict: Action frequency per episode
        cluster_action_dict: Per-agent action frequency per episode
        cluster_reward_dict: Per-agent cumulative reward per episode
        scatter_actions_dict: Action frequency for scatter agents
        scatter_action_dict: Per-agent action frequency for scatter agents
        scatter_reward_dict: Per-agent cumulative reward for scatter agents
        train_episodes: Number of training episodes
        train_log_every: Logging frequency
        alpha: Learning rate
        gamma: Discount factor
        decay_type: Epsilon decay type ("log" or "linear")
        decay: Decay parameter
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        print_metrics: Metric printing frequency
        logger: Logger object
        visualizer: Optional visualizer

    Returns:
        Tuple of (updated_qtable, updated_mixing_net)
    """

    n_actions = env.actions_n()
    old_s = {}
    old_a = {}
    previous_actions = {}
    agents_num = env.cluster_learners + env.scatter_learners
    agent_ids_by_mode = _agent_ids_by_mode_from_env(env, agents_num)
    mixing_net = _ensure_mode_mixing_networks(mixing_net, agent_ids_by_mode)

    # Tracking dicts for mixed vs individual Q-values
    only_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    only_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}

    print("Start training (QMIX)...\n")

    for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour="red", position=0, leave=False):
        env.reset()

        for tick in tqdm(range(1, params["episode_ticks"] + 1), desc="TICKS", colour="green", position=1, leave=False):
            step_info = {}
            agent_actions = []  # Track actions for all agents this tick

            for agent in env.agent_iter(max_iter=agents_num):
                cur_state, reward, _, _, _ = env.last(agent)
                agent_id = int(agent)
                cur_s = env.convert_observation(cur_state)

                if ep == 1 and tick == 1:
                    # First step: random action
                    action = np.random.randint(0, n_actions)
                else:
                    # Q-learning update for individual agent
                    old_value = qtable[agent_id, old_s[agent], old_a[agent]]
                    next_max = np.max(qtable[agent_id, cur_s])
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    qtable[agent_id, old_s[agent], old_a[agent]] = new_value

                    # Epsilon-greedy action selection
                    if random.uniform(0, 1) < epsilon:
                        action = np.random.randint(0, n_actions)
                    else:
                        action = int(np.argmax(qtable[agent_id, cur_s]))

                env.step(action)

                old_s[agent] = cur_s
                old_a[agent] = action
                agent_actions.append(action)

                step_info[agent_id] = {
                    "state": cur_s,
                    "observation": np.asarray(cur_state).copy(),
                    "reward": float(reward),
                    "current_action": int(action),
                    "previous_action": previous_actions.get(agent_id),
                    "position": env.learners[agent_id]["pos"],
                }
                previous_actions[agent_id] = int(action)

                # Track action/reward statistics
                if env.learners[agent_id]["mode"] == "c":
                    cluster_actions_dict[str(ep)][str(action)] += 1
                    cluster_action_dict[str(ep)][str(agent)][str(action)] += 1
                    cluster_reward_dict[str(ep)][str(agent)] += round(reward, 2)
                elif env.learners[agent_id]["mode"] == "s":
                    scatter_actions_dict[str(ep)][str(action)] += 1
                    scatter_action_dict[str(ep)][str(agent)][str(action)] += 1
                    scatter_reward_dict[str(ep)][str(agent)] += round(reward, 2)

            # QMIX value decomposition update: use one mixer per learner kind.
            # This happens after all agents have acted, allowing centralized
            # training without cross-mode information sharing.
            if tick > 1:  # Skip first tick (no old values to mix yet)
                _update_mode_mixing_networks(
                    qtable=qtable,
                    mixing_nets=mixing_net,
                    step_info=step_info,
                    old_s=old_s,
                    old_a=old_a,
                    agent_ids_by_mode=agent_ids_by_mode,
                    n_actions=n_actions,
                    gamma=gamma,
                )

            # Environment metrics per tick
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
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.fov,
                    env.ph_fov,
                )

        # Epsilon decay
        if decay_type == "log":
            epsilon = max(epsilon * decay, epsilon_min)
        elif decay_type == "linear":
            epsilon = max(epsilon - (1 - decay), epsilon_min)

        # Log metrics
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
                cluster_avg_rew = round(
                    (sum(cluster_reward_dict[str(ep)].values()) / params["episode_ticks"])
                    / params["cluster_learners"],
                    4,
                )
                value.append(cluster_avg_rew)
                value.extend(list(cluster_actions_dict[str(ep)].values()))
                tmp = [list(v.values()) for v in cluster_action_dict[str(ep)].values()]
                value.extend(list(itertools.chain(*tmp)))

            if params["scatter_learners"] > 0:
                scatter_avg_rew = round(
                    (sum(scatter_reward_dict[str(ep)].values()) / params["episode_ticks"])
                    / params["scatter_learners"],
                    4,
                )
                value.append(scatter_avg_rew)
                value.extend(list(scatter_actions_dict[str(ep)].values()))
                tmp = [list(v.values()) for v in scatter_action_dict[str(ep)].values()]
                value.extend(list(itertools.chain(*tmp)))

            value.append(round(epsilon, 4))
            logger.load_value(value)

            if ep % print_metrics == 0:
                print("\nMetrics (QMIX)")
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
    print("Training finished (QMIX)!\n")

    return qtable, mixing_net


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
    qtable: NDArray,
    test_log_every: int,
    logger,
    visualizer=None,
) -> None:
    """
    Evaluate QMIX agent(s) on the environment.

    In evaluation, each agent uses its learned Q-table for greedy action selection (decentralized execution).
    The mixing network is not used during evaluation (only for training).

    Args:
        env: The environment
        params: Environment configuration
        cluster_dict: Cumulative clustering metric
        cluster_actions_dict: Action frequency dict
        cluster_action_dict: Per-agent action frequency dict
        cluster_reward_dict: Per-agent cumulative reward dict
        scatter_actions_dict: Scatter action frequency
        scatter_action_dict: Per-agent scatter action frequency
        scatter_reward_dict: Per-agent scatter cumulative reward
        test_episodes: Number of evaluation episodes
        qtable: Learned Q-tables
        test_log_every: Logging frequency
        logger: Logger object
        visualizer: Optional visualizer
    """

    n_actions = env.actions_n()
    agents_num = env.cluster_learners + env.scatter_learners
    only_cluster_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    mixed_cluster_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    only_scatter_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}
    mixed_scatter_dict = {str(ep): 0.0 for ep in range(1, test_episodes + 1)}

    print("Start evaluation (QMIX)...\n")

    for ep in tqdm(range(1, test_episodes + 1), desc="EPISODES", colour="red", position=0, leave=False):
        env.reset()

        for tick in tqdm(
            range(1, params["episode_ticks"] + 1), desc="TICKS", colour="green", position=1, leave=False
        ):
            for agent in env.agent_iter(max_iter=agents_num):
                cur_state, reward, _, _, _ = env.last(agent)
                agent_id = int(agent)
                cur_s = env.convert_observation(cur_state)

                # Greedy action selection (no exploration)
                action = int(np.argmax(qtable[agent_id, cur_s]))

                env.step(action)

                # Track statistics
                if env.learners[agent_id]["mode"] == "c":
                    cluster_actions_dict[str(ep)][str(action)] += 1
                    cluster_action_dict[str(ep)][str(agent)][str(action)] += 1
                    cluster_reward_dict[str(ep)][str(agent)] += round(reward, 2)
                elif env.learners[agent_id]["mode"] == "s":
                    scatter_actions_dict[str(ep)][str(action)] += 1
                    scatter_action_dict[str(ep)][str(agent)][str(action)] += 1
                    scatter_reward_dict[str(ep)][str(agent)] += round(reward, 2)

            # Compute clustering metrics
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
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.fov,
                    env.ph_fov,
                )

        # Log metrics
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
                cluster_avg_rew = round(
                    (sum(cluster_reward_dict[str(ep)].values()) / params["episode_ticks"])
                    / params["cluster_learners"],
                    4,
                )
                value.append(cluster_avg_rew)
                value.extend(list(cluster_actions_dict[str(ep)].values()))
                tmp = [list(v.values()) for v in cluster_action_dict[str(ep)].values()]
                value.extend(list(itertools.chain(*tmp)))

            if params["scatter_learners"] > 0:
                scatter_avg_rew = round(
                    (sum(scatter_reward_dict[str(ep)].values()) / params["episode_ticks"])
                    / params["scatter_learners"],
                    4,
                )
                value.append(scatter_avg_rew)
                value.extend(list(scatter_actions_dict[str(ep)].values()))
                tmp = [list(v.values()) for v in scatter_action_dict[str(ep)].values()]
                value.extend(list(itertools.chain(*tmp)))

            logger.load_value(value)

    logger.empty_table()
    env.close()
    if visualizer is not None:
        visualizer.close()
    print("Evaluation finished (QMIX)!\n")
