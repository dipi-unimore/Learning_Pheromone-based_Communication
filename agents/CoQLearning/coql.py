import collections
import itertools
import random
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from agents.IQLearning import iql

_SHARED_INFORMATION_KEYS = ("observation", "action", "reward", "q_values")
_DEFAULT_SHARED_INFORMATION = {
    "observation": True,
    "action": True,
    "reward": True,
    "q_values": True,
}
_DEFAULT_COLLABORATION_CONFIG = {
    "enabled": True,
    "share_every_steps": 1,
    "recipient_selector": "all",
    "nearby_radius": 1,
    "share_rate": 0.1,
    "reward_weight": 1.0,
}


def create_agent(
    params: Dict,
    l_params: Dict,
    n_obs,
    n_actions,
    train,
) -> Tuple:
    """
    Keep the CoQL runner API aligned with IQL.
    """

    return iql.create_agent(params, l_params, n_obs, n_actions, train)



def resolve_collaboration_config(l_params: Dict) -> Dict:
    """
    Normalize collaboration settings and apply defaults.
    """

    collaboration = dict(_DEFAULT_COLLABORATION_CONFIG)
    collaboration.update(l_params.get("collaboration", {}))

    shared_information = dict(_DEFAULT_SHARED_INFORMATION)
    shared_information.update(collaboration.get("shared_information", {}))
    collaboration["shared_information"] = {
        key: bool(shared_information.get(key, False)) for key in _SHARED_INFORMATION_KEYS
    }

    collaboration["enabled"] = bool(collaboration.get("enabled", True))
    collaboration["share_every_steps"] = max(1, int(collaboration.get("share_every_steps", 1)))
    collaboration["nearby_radius"] = max(0, int(collaboration.get("nearby_radius", 1)))
    collaboration["share_rate"] = float(collaboration.get("share_rate", 0.1))
    collaboration["reward_weight"] = float(collaboration.get("reward_weight", 1.0))
    assert 0.0 <= collaboration["share_rate"] <= 1.0, "[ERROR] collaboration share_rate must be between 0 and 1"

    recipient_selector = collaboration.get("recipient_selector", "all")
    assert recipient_selector in {
        "all",
        "nearby",
        "similar",
    }, "[ERROR] collaboration recipient_selector must be one of: all, nearby, similar"

    return collaboration



def should_share_information(collaboration: Dict, tick: int) -> bool:
    """
    Return True when the configured collaboration step interval is met.
    """

    return (
        collaboration["enabled"]
        and any(collaboration["shared_information"].values())
        and tick % collaboration["share_every_steps"] == 0
    )



def _select_recipient_agents(
    recipient_id: int,
    env,
    step_info: Dict[int, Dict],
    collaboration: Dict,
) -> List[int]:
    """
    Select the peer agents whose information can be shared with the recipient.
    """

    selector = collaboration["recipient_selector"]
    recipient_mode = env.learners[recipient_id]["mode"]
    peer_ids = [
        agent_id
        for agent_id in step_info.keys()
        if agent_id != recipient_id and env.learners[agent_id]["mode"] == recipient_mode
    ]

    if selector == "all":
        return peer_ids

    if selector == "similar":
        recipient_observation = np.asarray(step_info[recipient_id]["observation"])
        return [
            agent_id
            for agent_id in peer_ids
            if np.array_equal(np.asarray(step_info[agent_id]["observation"]), recipient_observation)
        ]

    if selector == "nearby":
        recipient_position = step_info[recipient_id]["position"]
        radius = collaboration["nearby_radius"]
        return [
            agent_id
            for agent_id in peer_ids
            if _within_wrapped_square_radius(recipient_position, step_info[agent_id]["position"], env, radius)
        ]

    return []



def _within_wrapped_square_radius(source_pos: Tuple[int, int], target_pos: Tuple[int, int], env, radius: int) -> bool:
    """
    Check whether two positions are within a toroidal square radius measured in patches.
    """

    patch_size = env.patch_size
    width = env.W * patch_size
    height = env.H * patch_size
    max_distance = radius * patch_size

    dx = abs(source_pos[0] - target_pos[0])
    dy = abs(source_pos[1] - target_pos[1])
    dx = min(dx, width - dx)
    dy = min(dy, height - dy)

    return dx <= max_distance and dy <= max_distance



def _blend_row(current_row: NDArray, shared_row: NDArray, share_rate: float) -> NDArray:
    return (1 - share_rate) * current_row + share_rate * shared_row



def apply_collaboration(
    qtable: NDArray,
    env,
    step_info: Dict[int, Dict],
    collaboration: Dict,
    alpha: float,
    gamma: float,
) -> NDArray:
    """
    Apply a collaborative learning step without changing the tabular state space.
    """

    if not step_info:
        return qtable

    share_rate = collaboration["share_rate"]
    reward_weight = collaboration["reward_weight"]
    shared_information = collaboration["shared_information"]
    source_qtable = qtable.copy()

    for recipient_id in sorted(step_info.keys()):
        peer_ids = _select_recipient_agents(recipient_id, env, step_info, collaboration)
        if not peer_ids:
            continue

        recipient_state = step_info[recipient_id]["state"]
        recipient_action = step_info[recipient_id]["current_action"]

        if shared_information["observation"]:
            peer_states = [step_info[peer_id]["state"] for peer_id in peer_ids]
            shared_row = np.mean([source_qtable[recipient_id, peer_state] for peer_state in peer_states], axis=0)
            qtable[recipient_id, recipient_state] = _blend_row(
                qtable[recipient_id, recipient_state],
                shared_row,
                share_rate,
            )

        if shared_information["action"]:
            shared_actions = []
            for peer_id in peer_ids:
                previous_action = step_info[peer_id]["previous_action"]
                if previous_action is not None:
                    shared_actions.append(previous_action)
                shared_actions.append(step_info[peer_id]["current_action"])
            if shared_actions:
                majority_action = collections.Counter(shared_actions).most_common(1)[0][0]
                qtable[recipient_id, recipient_state, majority_action] += share_rate

        if shared_information["reward"]:
            shared_reward = float(np.mean([step_info[peer_id]["reward"] for peer_id in peer_ids]))
            old_value = source_qtable[recipient_id, recipient_state, recipient_action]
            next_max = np.max(source_qtable[recipient_id, recipient_state])
            shared_target = (1 - alpha) * old_value + alpha * ((reward_weight * shared_reward) + gamma * next_max)
            qtable[recipient_id, recipient_state, recipient_action] = (
                (1 - share_rate) * old_value + share_rate * shared_target
            )

        if shared_information["q_values"]:
            shared_q_row = np.mean(
                [source_qtable[peer_id, step_info[peer_id]["state"]] for peer_id in peer_ids],
                axis=0,
            )
            qtable[recipient_id, recipient_state] = _blend_row(
                qtable[recipient_id, recipient_state],
                shared_q_row,
                share_rate,
            )

    return qtable



def train(
    env,
    params: Dict,
    l_params: Dict,
    qtable: NDArray,
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
) -> NDArray:
    """
    Training function with configurable collaboration.
    """

    collaboration = resolve_collaboration_config(l_params)
    n_actions = env.actions_n()
    old_s = {}
    old_a = {}
    previous_actions = {}
    agents_num = env.cluster_learners + env.scatter_learners
    only_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_cluster_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    only_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}
    mixed_scatter_dict = {str(ep): 0.0 for ep in range(1, train_episodes + 1)}

    print("Start training...\n")

    for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour="red", position=0, leave=False):
        env.reset()

        for tick in tqdm(range(1, params["episode_ticks"] + 1), desc="TICKS", colour="green", position=1, leave=False):
            step_info = {}
            for agent in env.agent_iter(max_iter=agents_num):
                cur_state, reward, _, _, _ = env.last(agent)
                agent_id = int(agent)
                cur_s = env.convert_observation(cur_state)

                if ep == 1 and tick == 1:
                    action = np.random.randint(0, n_actions)
                else:
                    old_value = qtable[agent_id, old_s[agent], old_a[agent]]
                    next_max = np.max(qtable[agent_id, cur_s])
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    qtable[agent_id, old_s[agent], old_a[agent]] = new_value

                    if random.uniform(0, 1) < epsilon:
                        action = np.random.randint(0, n_actions)
                    else:
                        action = np.argmax(qtable[agent_id][cur_s])

                env.step(action)

                old_s[agent] = cur_s
                old_a[agent] = action

                step_info[agent_id] = {
                    "state": cur_s,
                    "observation": np.asarray(cur_state).copy(),
                    "reward": float(reward),
                    "current_action": int(action),
                    "previous_action": previous_actions.get(agent_id),
                    "position": env.learners[agent_id]["pos"],
                }
                previous_actions[agent_id] = int(action)

                if env.learners[agent_id]["mode"] == "c":
                    cluster_actions_dict[str(ep)][str(action)] += 1
                    cluster_action_dict[str(ep)][str(agent)][str(action)] += 1
                    cluster_reward_dict[str(ep)][str(agent)] += round(reward, 2)
                elif env.learners[agent_id]["mode"] == "s":
                    scatter_actions_dict[str(ep)][str(action)] += 1
                    scatter_action_dict[str(ep)][str(agent)][str(action)] += 1
                    scatter_reward_dict[str(ep)][str(agent)] += round(reward, 2)

            if should_share_information(collaboration, tick):
                qtable = apply_collaboration(qtable, env, step_info, collaboration, alpha, gamma)

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
                print("\nMetrics ")
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
    print("Training finished!\n")

    return qtable



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
    Evaluation remains identical to IQL for backward compatibility.
    """

    return iql.eval(
        env,
        params,
        cluster_dict,
        cluster_actions_dict,
        cluster_action_dict,
        cluster_reward_dict,
        scatter_actions_dict,
        scatter_action_dict,
        scatter_reward_dict,
        test_episodes,
        qtable,
        test_log_every,
        logger,
        visualizer,
    )


