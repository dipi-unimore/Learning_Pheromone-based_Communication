import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
from typing import Dict, Tuple

def policy(state: NDArray, threshold: float) -> int:
    """
    Choose an action using a deterministic policy.
    """

    if np.all(state > threshold):
        action = 5
    else:
        action = 4
    return action

def run(env, params: Dict, episodes: int, visualizer=None) -> Tuple:
    """
    Run the simulation.
    """

    assert(params["scatter_learners"] == 0), "Only clustering agents are allowed, scattering agents must be 0!"
    learner_population = params['cluster_learners']
    
    # metrics data structs 
    reward_dict = {
        str(ep): {
            str(ag): 0 
            for ag in range(learner_population)
        }
        for ep in range(1, episodes + 1)
    }
    cluster_dict = {str(ep): 0.0 for ep in range(1, episodes + 1)}
    avg_reward_dict = []
    avg_cluster_dict = []

    print("Start running...\n")
        # follow_pheromone e se sei in un cluster lay_pheromone
    for ep in tqdm(range(1, episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', position=1, leave=False):
            for agent in env.agent_iter(max_iter=learner_population):
                cur_state, reward, _, _, _ = env.last(agent)
                action = policy(cur_state, env.sniff_threshold)
                env.step(action)
            
                reward_dict[str(ep)][str(agent)] += round(reward, 2)
            
            cluster_dict[str(ep)] += round(env.avg_cluster(), 2) 
            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.learners,
                    env.fov,
                    env.ph_fov
                )
        
        avg_rew = round((sum(reward_dict[str(ep)].values()) / params["episode_ticks"]) / learner_population, 4)
        avg_reward_dict.append(avg_rew)
        avg_cluster = round(cluster_dict[str(ep)] / params["episode_ticks"], 2)
        avg_cluster_dict.append(avg_cluster)

    env.close()
    if visualizer != None:
        visualizer.close()
    print("Training finished!\n")

    return avg_reward_dict, avg_cluster_dict