from typing import Any, Dict

from agents.utils.epymarl_launcher import run_epymarl_command

ALGORITHM_NAME = "qmix"


def train(
    l_params: Dict,
    seed: int,
    env_params_path: str,
    learning_params_path: str,
    logger_params_path: str,
    run_tag: str,
) -> Dict[str, Any]:
    return run_epymarl_command(
        l_params=l_params,
        algorithm=ALGORITHM_NAME,
        train=True,
        seed=seed,
        env_params_path=env_params_path,
        learning_params_path=learning_params_path,
        logger_params_path=logger_params_path,
        run_tag=run_tag,
    )


def eval(
    l_params: Dict,
    seed: int,
    env_params_path: str,
    learning_params_path: str,
    logger_params_path: str,
    run_tag: str,
) -> Dict[str, Any]:
    return run_epymarl_command(
        l_params=l_params,
        algorithm=ALGORITHM_NAME,
        train=False,
        seed=seed,
        env_params_path=env_params_path,
        learning_params_path=learning_params_path,
        logger_params_path=logger_params_path,
        run_tag=run_tag,
    )
