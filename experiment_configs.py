import os
import re
from typing import Dict, List, Optional


EXPERIMENT_FILE_BASENAMES = {
    "params_path": "env-params",
    "learning_params_path": "learning-params",
    "visualizer_params_path": "env_visualizer-params",
    "logger_params_path": "logger-params",
}


DEFAULT_CONFIG_PATHS = {
    "params_path": "environments/slime/config/env-params.json",
    "learning_params_path": "agents/IQLearning/config/learning-params.json",
    "visualizer_params_path": "environments/slime/config/env_visualizer-params.json",
    "logger_params_path": "agents/IQLearning/config/logger-params.json",
}


_EXPERIMENT_FILE_PATTERN = re.compile(
    r"^(learning-params|logger-params|env-params|env_visualizer-params)-(\d+)\.json$"
)


def collect_experiment_ids(experiments_dir: str) -> List[int]:
    """
    Return sorted experiment ids found in the given directory.
    """

    if not experiments_dir or not os.path.isdir(experiments_dir):
        return []

    experiment_ids = set()
    for file_name in os.listdir(experiments_dir):
        match = _EXPERIMENT_FILE_PATTERN.match(file_name)
        if match:
            experiment_ids.add(int(match.group(2)))

    return sorted(experiment_ids)


def resolve_experiment_paths(
    experiment_id: int,
    experiments_dir: str,
    default_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Resolve config paths for a single experiment id, falling back to defaults.
    """

    resolved_paths = dict(DEFAULT_CONFIG_PATHS if default_paths is None else default_paths)

    for arg_name, base_name in EXPERIMENT_FILE_BASENAMES.items():
        experiment_path = os.path.join(experiments_dir, f"{base_name}-{experiment_id}.json")
        if os.path.isfile(experiment_path):
            resolved_paths[arg_name] = experiment_path

    return resolved_paths


def resolve_random_seeds(random_seed: int, random_seeds: Optional[List[int]] = None) -> List[int]:
    """
    Resolve active seeds while preserving backward compatibility with a single seed.
    """

    return list(random_seeds) if random_seeds else [random_seed]


def build_run_tag(seed: int, experiment_id: Optional[int] = None) -> str:
    """
    Build a human-readable run tag for output traceability.
    """

    if experiment_id is None:
        return f"seed_{seed}"
    return f"exp_{experiment_id}_seed_{seed}"


