from agents.MAPPOLearning import mappo
from slime_epymarl import run_cli


if __name__ == "__main__":
    run_cli(
        runner=mappo,
        default_learning_params_path="agents/MAPPOLearning/config/learning-params.json",
        default_logger_params_path="agents/MAPPOLearning/config/logger-params.json",
        description="Run Slime experiments with MAPPO via EpyMARL",
    )

