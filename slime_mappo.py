import argparse
import datetime
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch

from agents.MAPPOLearning import mappo
from agents.utils.logger import Logger
from environments.slime.slime import Slime
from experiment_configs import build_run_tag, collect_experiment_ids, resolve_experiment_paths, resolve_random_seeds


def read_params(params_path: str, learning_params_path: str, visualizer_params_path: str, logger_params_path: str) -> Tuple:
    params, l_params, v_params, log_params = dict(), dict(), dict(), dict()
    for path, target, label in [
        (learning_params_path, l_params, "learning"),
        (params_path, params, "params"),
        (visualizer_params_path, v_params, "visualizer"),
        (logger_params_path, log_params, "logger"),
    ]:
        try:
            with open(path) as f:
                target.update(json.load(f))
        except Exception as e:
            print(f"[ERROR] could not open {label} params file: {e}")
    return params, l_params, v_params, log_params


def create_logger(curdir: str, params: Dict, l_params: Dict, log_params: Dict, train: bool, weights_path=None, run_tag: str = "") -> Tuple:
    log_every = log_params["train_log_every"] if train else log_params["test_log_every"]
    log = Logger(
        curdir,
        params,
        l_params,
        log_params,
        train=train,
        buffer_size=log_params["buffer_size"],
        weights_file=weights_path,
        run_tag=run_tag,
    )
    return log, log_every


def main(args) -> None:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    curdir = os.path.dirname(os.path.abspath(__file__))
    params, l_params, v_params, log_params = read_params(
        args.params_path,
        args.learning_params_path,
        args.visualizer_params_path,
        args.logger_params_path,
    )
    env = Slime(args.random_seed, **params)
    env_vis = None
    if args.render:
        from environments.slime.slime import SlimeVisualizer
        env_vis = SlimeVisualizer(env.W_pixels, env.H_pixels, **v_params)

    n_obs, n_actions = env.observations_n(), env.actions_n()
    run_tag = getattr(args, "run_tag", "")

    if args.train:
        (
            actor_nets,
            critics,
            device,
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
        ) = mappo.create_agent(params, l_params, n_obs, n_actions, True)
        del device

        logger, train_log_every = create_logger(curdir, params, l_params, log_params, True, run_tag=run_tag)
        start = datetime.datetime.now()
        actor_nets, critics = mappo.train(
            env,
            params,
            l_params,
            actor_nets,
            critics,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
            train_episodes,
            train_log_every,
            actor_lr,
            critic_lr,
            gamma,
            args.print_metrics,
            logger,
            env_vis,
        )
        end = datetime.datetime.now()
        logger.save_computation_time(end - start)
        print(f"Training time: {end - start}\n")
        logger.save_model(mappo.pack_model(actor_nets, critics))
    else:
        (
            actor_nets,
            critics,
            device,
            test_episodes,
            cluster_dict,
            cluster_actions_dict,
            cluster_action_dict,
            cluster_reward_dict,
            scatter_actions_dict,
            scatter_action_dict,
            scatter_reward_dict,
        ) = mappo.create_agent(params, l_params, n_obs, n_actions, False)
        del device

        logger, test_log_every = create_logger(curdir, params, l_params, log_params, False, args.qtable_path, run_tag=run_tag)
        weights = logger.load_model()
        mappo.unpack_model(weights, actor_nets, critics)
        start = datetime.datetime.now()
        mappo.eval(
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
            actor_nets,
            test_log_every,
            logger,
            env_vis,
            l_params=l_params,
        )
        end = datetime.datetime.now()
        logger.save_computation_time(end - start, train=False)
        print(f"\nTesting time: {end - start}")


def run_experiments_sequence(args) -> None:
    experiment_ids = collect_experiment_ids(args.experiments_dir)
    assert experiment_ids, "[ERROR] no experiment configuration files found in experiments directory"
    default_paths = {k: getattr(args, k) for k in ["params_path", "learning_params_path", "visualizer_params_path", "logger_params_path"]}
    for experiment_id in experiment_ids:
        experiment_paths = resolve_experiment_paths(experiment_id, args.experiments_dir, default_paths)
        for seed in resolve_random_seeds(args.random_seed, args.random_seeds):
            experiment_args = argparse.Namespace(**vars(args))
            experiment_args.params_path = experiment_paths["params_path"]
            experiment_args.learning_params_path = experiment_paths["learning_params_path"]
            experiment_args.visualizer_params_path = experiment_paths["visualizer_params_path"]
            experiment_args.logger_params_path = experiment_paths["logger_params_path"]
            experiment_args.random_seed = seed
            experiment_args.run_tag = build_run_tag(seed=seed, experiment_id=experiment_id)
            print(f"\n[EXPERIMENT {experiment_id}] [SEED {seed}] Current args: {experiment_args}")
            if check_args(experiment_args):
                main(experiment_args)


def run_single_experiment(args) -> None:
    for seed in resolve_random_seeds(args.random_seed, args.random_seeds):
        run_args = argparse.Namespace(**vars(args))
        run_args.random_seed = seed
        run_args.run_tag = build_run_tag(seed=seed)
        print(f"\n[SEED {seed}] Current args: {run_args}")
        if check_args(run_args):
            main(run_args)


def check_args(args) -> bool:
    for path_name in ["params_path", "learning_params_path", "visualizer_params_path", "logger_params_path"]:
        path = getattr(args, path_name)
        assert path and os.path.isfile(path) and path.endswith(".json"), f"[ERROR] {path_name} is empty or is not a json file"
    if args.qtable_path is not None:
        assert args.qtable_path.endswith(".npy"), "[ERROR] model weights file must be a .npy file"
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, default="environments/slime/config/env-params.json", required=False)
    parser.add_argument("--visualizer_params_path", type=str, default="environments/slime/config/env_visualizer-params.json", required=False)
    parser.add_argument("--learning_params_path", type=str, default="agents/MAPPOLearning/config/learning-params.json", required=False)
    parser.add_argument("--logger_params_path", type=str, default="agents/MAPPOLearning/config/logger-params.json", required=False)
    parser.add_argument("--qtable_path", type=str, required=False, help="Saved .npy MAPPO model path used during evaluation")
    parser.add_argument("--train", type=bool, default=True, required=False)
    parser.add_argument("--random_seed", type=int, default=42, required=False)
    parser.add_argument("--random_seeds", "--random-seeds", type=int, nargs="+", default=None, required=False, help="Optional list of random seeds. If provided, repeats each run once per seed")
    parser.add_argument("--print_metrics", type=int, default=100, required=False)
    parser.add_argument("--render", type=bool, default=False, required=False)
    parser.add_argument("--experiments_dir", type=str, default="", required=False, help="Directory containing optional *-params-X.json files to run experiments sequentially")
    args = parser.parse_args()
    if args.experiments_dir:
        run_experiments_sequence(args)
    else:
        run_single_experiment(args)
