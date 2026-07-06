import shlex
import subprocess
from typing import Any, Dict


def _resolve_command_template(epymarl_config: Dict, train: bool) -> str:
    if train and epymarl_config.get("command_template_train"):
        return str(epymarl_config["command_template_train"])
    if (not train) and epymarl_config.get("command_template_eval"):
        return str(epymarl_config["command_template_eval"])
    if epymarl_config.get("command_template"):
        return str(epymarl_config["command_template"])
    return ""


def run_epymarl_command(
    l_params: Dict,
    algorithm: str,
    train: bool,
    seed: int,
    env_params_path: str,
    learning_params_path: str,
    logger_params_path: str,
    run_tag: str,
) -> Dict[str, Any]:
    """
    Execute an EpyMARL command template configured in the learning params.

    The template can be defined as command_template_train/command_template_eval
    or as a shared command_template under the `epymarl` block.
    """

    epymarl_config = dict(l_params.get("epymarl", {}))
    command_template = _resolve_command_template(epymarl_config, train)
    if not command_template or command_template == "__SET_ME__":
        raise ValueError(
            "[ERROR] missing EpyMARL command template. "
            "Set epymarl.command_template_train/command_template_eval in learning params."
        )

    placeholders = {
        "algorithm": algorithm,
        "seed": seed,
        "env_params_path": env_params_path,
        "learning_params_path": learning_params_path,
        "logger_params_path": logger_params_path,
        "run_tag": run_tag,
    }
    try:
        command = command_template.format(**placeholders)
    except KeyError as exc:
        raise ValueError(f"[ERROR] missing template placeholder: {exc}") from exc

    command_parts = shlex.split(command)
    process = subprocess.run(command_parts, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
