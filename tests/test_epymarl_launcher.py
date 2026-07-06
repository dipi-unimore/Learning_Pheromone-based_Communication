import unittest
from unittest import mock

from agents.MAPPOLearning import mappo
from agents.QMIXLearning import qmix
from agents.utils.epymarl_launcher import run_epymarl_command


class TestEpyMARLLauncher(unittest.TestCase):
    def test_missing_command_template_raises(self):
        """Test that missing command template raises ValueError when forcing subprocess."""
        with self.assertRaises(ValueError):
            run_epymarl_command(
                l_params={"epymarl": {"force_subprocess": True}},
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
        """Test that QMIX training substitutes {algorithm} correctly."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
        l_params = {
            "epymarl": {
                "force_subprocess": True,
                "command_template_train": "python -m launch --algo {algorithm} --seed {seed} --env {env_params_path} --tag {run_tag}",
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
        self.assertEqual(result["method"], "subprocess")
        called_command = mock_run.call_args.args[0]
        self.assertIn("qmix", called_command)
        self.assertIn("5", called_command)

    @mock.patch("agents.utils.epymarl_launcher.subprocess.run")
    def test_mappo_eval_uses_eval_template(self, mock_run):
        """Test that MAPPO evaluation uses the eval template."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="eval", stderr="")
        l_params = {
            "epymarl": {
                "force_subprocess": True,
                "command_template_eval": "python -m launch --algo {algorithm} --eval --seed {seed}",
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
        self.assertEqual(result["method"], "subprocess")

    @mock.patch("agents.utils.epymarl_launcher._try_direct_api_call")
    def test_direct_api_attempted_by_default(self, mock_direct_api):
        """Test that direct API is attempted when use_direct_api=True."""
        mock_direct_api.return_value = {
            "method": "direct_api",
            "command": "direct_api:qmix",
            "returncode": 0,
            "stdout": "[Direct API] Executed qmix",
            "stderr": "",
        }

        l_params = {
            "epymarl": {
                "use_direct_api": True,
                "force_subprocess": False,
            }
        }

        result = qmix.train(
            l_params=l_params,
            seed=10,
            env_params_path="environments/slime/config/env-params.json",
            learning_params_path="agents/QMIXLearning/config/learning-params.json",
            logger_params_path="agents/QMIXLearning/config/logger-params.json",
            run_tag="seed_10",
        )

        # When direct API returns a result, it should be used
        self.assertEqual(result["method"], "direct_api")
        self.assertTrue(result["command"].startswith("direct_api"))

    @mock.patch("agents.utils.epymarl_launcher.subprocess.run")
    def test_force_subprocess_ignores_direct_api(self, mock_run):
        """Test that force_subprocess=True bypasses direct API."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
        l_params = {
            "epymarl": {
                "force_subprocess": True,
                "command_template_train": "python test --algo {algorithm}",
            }
        }

        result = qmix.train(
            l_params=l_params,
            seed=20,
            env_params_path="env.json",
            learning_params_path="learning.json",
            logger_params_path="logger.json",
            run_tag="seed_20",
        )

        self.assertEqual(result["method"], "subprocess")
        self.assertFalse(result["command"].startswith("direct_api"))


if __name__ == "__main__":
    unittest.main()

