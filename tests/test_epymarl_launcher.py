import unittest
from unittest import mock

from agents.MAPPOLearning import mappo
from agents.QMIXLearning import qmix
from agents.utils.epymarl_launcher import run_epymarl_command


class TestEpyMARLLauncher(unittest.TestCase):
    def test_missing_command_template_raises(self):
        with self.assertRaises(ValueError):
            run_epymarl_command(
                l_params={},
                algorithm="qmix",
                train=True,
                seed=7,
                env_params_path="env.json",
                learning_params_path="learning.json",
                logger_params_path="logger.json",
                run_tag="seed_7",
            )

    @mock.patch("agents.utils.epymarl_launcher.subprocess.run")
    def test_qmix_train_uses_algorithm_placeholder(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
        l_params = {
            "epymarl": {
                "command_template_train": "python -m launch --algo {algorithm} --seed {seed} --env {env_params_path} --tag {run_tag}"
            }
        }

        result = qmix.train(
            l_params=l_params,
            seed=5,
            env_params_path="environments/slime/config/env-params.json",
            learning_params_path="agents/QMIXLearning/config/learning-params.json",
            logger_params_path="agents/QMIXLearning/config/logger-params.json",
            run_tag="seed_5",
        )

        self.assertEqual(result["returncode"], 0)
        self.assertIn("--algo qmix", result["command"])
        called_command = mock_run.call_args.args[0]
        self.assertIn("qmix", called_command)
        self.assertIn("5", called_command)

    @mock.patch("agents.utils.epymarl_launcher.subprocess.run")
    def test_mappo_eval_uses_eval_template(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0, stdout="eval", stderr="")
        l_params = {
            "epymarl": {
                "command_template_eval": "python -m launch --algo {algorithm} --eval --seed {seed}"
            }
        }

        result = mappo.eval(
            l_params=l_params,
            seed=12,
            env_params_path="environments/slime/config/env-params.json",
            learning_params_path="agents/MAPPOLearning/config/learning-params.json",
            logger_params_path="agents/MAPPOLearning/config/logger-params.json",
            run_tag="seed_12",
        )

        self.assertEqual(result["returncode"], 0)
        self.assertIn("--algo mappo", result["command"])
        self.assertIn("--eval", result["command"])


if __name__ == "__main__":
    unittest.main()

