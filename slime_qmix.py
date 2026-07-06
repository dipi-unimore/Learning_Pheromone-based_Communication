from agents.QMIXLearning import qmix
from slime_epymarl import run_cli


if __name__ == "__main__":
    run_cli(
        runner=qmix,
        default_learning_params_path="agents/QMIXLearning/config/learning-params.json",
        default_logger_params_path="agents/QMIXLearning/config/logger-params.json",
        description="Run Slime experiments with QMIX via EpyMARL",
    )

