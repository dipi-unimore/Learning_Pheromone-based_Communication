import os
import tempfile
import unittest

from agents.utils.logger import Logger


class TestLoggerTraceability(unittest.TestCase):
    def test_logger_paths_include_run_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {
                "cluster_learners": 1,
                "scatter_learners": 0,
                "actions": ["random-walk"],
            }
            l_params = {}
            log_params = {
                "train_log_every": 1,
                "test_log_every": 1,
                "buffer_size": 1,
                "train_output_file": "train-output",
                "train_params_file": "train-params",
                "train_weights_file": "train-weights",
                "test_output_file": "test-output",
                "test_params_file": "test-params",
                "test_weights_file": "test-weights",
            }

            logger = Logger(
                curdir=tmpdir,
                params=params,
                l_params=l_params,
                log_params=log_params,
                train=True,
                buffer_size=1,
                run_tag="exp_2_seed_99",
            )

            self.assertIn("exp_2_seed_99", logger.output_file)
            self.assertIn("exp_2_seed_99", logger.params_file)
            self.assertIn("exp_2_seed_99", logger.weights_file)
            self.assertTrue(os.path.isfile(logger.params_file))


if __name__ == "__main__":
    unittest.main()

