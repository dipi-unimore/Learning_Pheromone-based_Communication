import argparse
import json
import os
from typing import Dict

from experiment_configs import (
    build_run_tag,
    collect_experiment_ids,
    resolve_experiment_paths,
    resolve_random_seeds,
)


def _read_json_file(path: str) -> Dict:
    with open(path) as file:
        return json.load(file)


def check_args(args) -> bool:
    assert (
        args.params_path != ""
        and os.path.isfile(args.params_path)
        and args.params_path.endswith(".json")
    ), "[ERROR] params path is empty or is not a file or is not a json file"

    assert (
        args.learning_params_path != ""
        and os.path.isfile(args.learning_params_path)
        and args.learning_params_path.endswith(".json")
    ), "[ERROR] learning params path is empty or is not a file or is not a json file"

    assert (
        args.visualizer_params_path != ""
        and os.path.isfile(args.visualizer_params_path)
        and args.visualizer_params_path.endswith(".json")
    ), "[ERROR] visualizer params path is empty or is not a file or is not a json file"

    assert (
        args.logger_params_path != ""
        and os.path.isfile(args.logger_params_path)
        and args.logger_params_path.endswith(".json")
    ), "[ERROR] logger params path is empty or is not a file or is not a json file"

    return True


def _run_once(args, runner) -> None:
    l_params = _read_json_file(args.learning_params_path)
    run_tag = getattr(args, "run_tag", "")

    if args.train:
        result = runner.train(
            l_params=l_params,
            seed=args.random_seed,
            env_params_path=args.params_path,
            learning_params_path=args.learning_params_path,
            logger_params_path=args.logger_params_path,
            run_tag=run_tag,
        )
    else:
        result = runner.eval(
            l_params=l_params,
            seed=args.random_seed,
            env_params_path=args.params_path,
            learning_params_path=args.learning_params_path,
            logger_params_path=args.logger_params_path,
            run_tag=run_tag,
        )

    # Report execution method (direct API vs subprocess)
    method = result.get("method", "unknown")
    print(f"\n[EXECUTION METHOD] {method.upper()}")
    print(f"Command: {result['command']}")
    if result["stdout"]:
        print(f"STDOUT:\n{result['stdout']}")
    if result["stderr"]:
        print(f"STDERR:\n{result['stderr']}")
    if result["returncode"] != 0:
        raise RuntimeError(
            f"[ERROR] EpyMARL execution failed with return code {result['returncode']} "
            f"(method: {method})"
        )


def run_experiments_sequence(args, runner) -> None:
    experiment_ids = collect_experiment_ids(args.experiments_dir)
    assert experiment_ids, "[ERROR] no experiment configuration files found in experiments directory"

    default_paths = {
        "params_path": args.params_path,
        "learning_params_path": args.learning_params_path,
        "visualizer_params_path": args.visualizer_params_path,
        "logger_params_path": args.logger_params_path,
    }

    seeds = resolve_random_seeds(args.random_seed, args.random_seeds)
    for experiment_id in experiment_ids:
        experiment_paths = resolve_experiment_paths(experiment_id, args.experiments_dir, default_paths)
        for seed in seeds:
            experiment_args = argparse.Namespace(**vars(args))
            experiment_args.params_path = experiment_paths["params_path"]
            experiment_args.learning_params_path = experiment_paths["learning_params_path"]
            experiment_args.visualizer_params_path = experiment_paths["visualizer_params_path"]
            experiment_args.logger_params_path = experiment_paths["logger_params_path"]
            experiment_args.random_seed = seed
            experiment_args.run_tag = build_run_tag(seed=seed, experiment_id=experiment_id)

            print(f"\n[EXPERIMENT {experiment_id}] [SEED {seed}] Current args: {experiment_args}")
            if check_args(experiment_args):
                _run_once(experiment_args, runner)


def run_single_experiment(args, runner) -> None:
    seeds = resolve_random_seeds(args.random_seed, args.random_seeds)
    for seed in seeds:
        run_args = argparse.Namespace(**vars(args))
        run_args.random_seed = seed
        run_args.run_tag = build_run_tag(seed=seed)

        print(f"\n[SEED {seed}] Current args: {run_args}")
        if check_args(run_args):
            _run_once(run_args, runner)


def run_cli(
    runner,
    default_learning_params_path: str,
    default_logger_params_path: str,
    description: str,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--params_path",
        type=str,
        default="environments/slime/config/env-params.json",
        required=False,
    )
    parser.add_argument(
        "--visualizer_params_path",
        type=str,
        default="environments/slime/config/env_visualizer-params.json",
        required=False,
    )
    parser.add_argument(
        "--learning_params_path",
        type=str,
        default=default_learning_params_path,
        required=False,
    )
    parser.add_argument(
        "--logger_params_path",
        type=str,
        default=default_logger_params_path,
        required=False,
    )
    parser.add_argument("--train", type=bool, default=True, required=False)
    parser.add_argument("--random_seed", type=int, default=42, required=False)
    parser.add_argument(
        "--random_seeds",
        "--random-seeds",
        type=int,
        nargs="+",
        default=None,
        required=False,
        help="Optional list of random seeds. If provided, repeats each run once per seed",
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="",
        required=False,
        help="Directory containing optional *-params-X.json files to run experiments sequentially",
    )

    args = parser.parse_args()
    if args.experiments_dir:
        run_experiments_sequence(args, runner)
    else:
        run_single_experiment(args, runner)

