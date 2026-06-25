import os
import tempfile
import unittest

from experiment_configs import (
    build_run_tag,
    collect_experiment_ids,
    resolve_experiment_paths,
    resolve_random_seeds,
)


class TestExperimentConfigs(unittest.TestCase):
    def test_collect_experiment_ids_sorted_and_unique(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for file_name in [
                "env-params-2.json",
                "learning-params-1.json",
                "logger-params-2.json",
                "env_visualizer-params-1.json",
                "random.txt",
                "env-params-A.json",
            ]:
                with open(os.path.join(tmpdir, file_name), "w", encoding="utf-8") as file:
                    file.write("{}")

            self.assertEqual(collect_experiment_ids(tmpdir), [1, 2])

    def test_resolve_experiment_paths_uses_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            default_paths = {
                "params_path": "default-env.json",
                "learning_params_path": "default-learning.json",
                "visualizer_params_path": "default-visualizer.json",
                "logger_params_path": "default-logger.json",
            }

            env_override = os.path.join(tmpdir, "env-params-4.json")
            with open(env_override, "w", encoding="utf-8") as file:
                file.write("{}")

            resolved_paths = resolve_experiment_paths(4, tmpdir, default_paths)

            self.assertEqual(resolved_paths["params_path"], env_override)
            self.assertEqual(resolved_paths["learning_params_path"], "default-learning.json")
            self.assertEqual(resolved_paths["visualizer_params_path"], "default-visualizer.json")
            self.assertEqual(resolved_paths["logger_params_path"], "default-logger.json")

    def test_resolve_random_seeds_prefers_list(self):
        self.assertEqual(resolve_random_seeds(42, [10, 20, 30]), [10, 20, 30])

    def test_resolve_random_seeds_falls_back_to_single_seed(self):
        self.assertEqual(resolve_random_seeds(42, None), [42])

    def test_build_run_tag_includes_experiment_and_seed(self):
        self.assertEqual(build_run_tag(seed=99, experiment_id=3), "exp_3_seed_99")

    def test_build_run_tag_seed_only(self):
        self.assertEqual(build_run_tag(seed=99), "seed_99")


if __name__ == "__main__":
    unittest.main()

