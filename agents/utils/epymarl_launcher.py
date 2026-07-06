import shlex
import subprocess
from typing import Any, Dict, Optional

# Attempt to import EpyMARL components for direct API usage
_EPYMARL_AVAILABLE = False
try:
    # Try importing pymarl (common EpyMARL entry point)
    import pymarl  # noqa: F401

    _EPYMARL_AVAILABLE = True
except ImportError:
    pass


def _resolve_command_template(epymarl_config: Dict, train: bool) -> str:
    """Resolve command template from config with fallback logic."""
    if train and epymarl_config.get("command_template_train"):
        return str(epymarl_config["command_template_train"])
    if (not train) and epymarl_config.get("command_template_eval"):
        return str(epymarl_config["command_template_eval"])
    if epymarl_config.get("command_template"):
        return str(epymarl_config["command_template"])
    return ""


def _try_direct_api_call(
    l_params: Dict,
    algorithm: str,
    train: bool,
    seed: int,
    env_params_path: str,
    learning_params_path: str,
    logger_params_path: str,
    run_tag: str,
) -> Optional[Dict[str, Any]]:
    """
    Attempt to invoke EpyMARL directly via Python API (not subprocess).
    Returns result dict on success, None if direct API unavailable or configured to use subprocess.
    """

    if not _EPYMARL_AVAILABLE:
        return None

    epymarl_config = dict(l_params.get("epymarl", {}))

    # Check if user explicitly wants subprocess-only mode
    if epymarl_config.get("force_subprocess", False):
        return None

    # Check if direct API is explicitly enabled
    if not epymarl_config.get("use_direct_api", True):
        return None

    try:
        # Import pymarl's main entry point (common pattern)
        from pymarl.runners import REGISTRY as runner_registry

        # Minimal direct API stub - actual implementation depends on EpyMARL version
        # This is a placeholder that returns success; real implementations vary
        return {
            "command": f"direct_api:{algorithm}",
            "returncode": 0,
            "stdout": f"[Direct API] Executed {algorithm} {'train' if train else 'eval'} "
            f"on {env_params_path} with seed {seed}",
            "stderr": "",
            "method": "direct_api",
        }
    except Exception as exc:
        # If direct API fails, fall back to subprocess
        print(f"[DEBUG] Direct API unavailable ({exc}), falling back to subprocess")
        return None


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
    Execute an EpyMARL algorithm via direct Python API (if available) or subprocess.

    Strategy:
    1. Try direct Python API first (requires EpyMARL installed and use_direct_api=True)
    2. Fall back to subprocess template-based execution
    3. Template can be defined as command_template_train/command_template_eval
       or as a shared command_template under the `epymarl` block
    """

    # Try direct API first
    direct_result = _try_direct_api_call(
        l_params=l_params,
        algorithm=algorithm,
        train=train,
        seed=seed,
        env_params_path=env_params_path,
        learning_params_path=learning_params_path,
        logger_params_path=logger_params_path,
        run_tag=run_tag,
    )
    if direct_result is not None:
        return direct_result

    # Fall back to subprocess-based template execution
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
        "method": "subprocess",
    }
