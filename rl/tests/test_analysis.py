from __future__ import annotations

import argparse
import math
import unittest

import numpy as np

from rl.analysis.linear_probe import fit_episode_ridge_probe, split_episode_ids
from rl.analysis.metrics import action_metrics, aggregate_episodes, degradation
from rl.analysis.robustness import robustness_degradation
from rl.analysis.stability import central_difference_jacobian
from rl.analysis.statistics import aggregate_training_seeds, mean_confidence_interval, wilson_interval
from rl.scripts._common import add_environment_arguments, environment_overrides


class ActionMetricTests(unittest.TestCase):
    def test_rms_action_delta_and_jerk(self):
        result = action_metrics(np.asarray([[0.0], [1.0], [3.0]]))
        self.assertAlmostEqual(result["action_rms"], math.sqrt(10.0 / 3.0))
        self.assertAlmostEqual(result["action_delta_rms"], math.sqrt(2.5))
        self.assertAlmostEqual(result["action_jerk_rms"], 1.0)


class EnvironmentArgumentTests(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        add_environment_arguments(self.parser)

    def test_partial_observation_flags_enable_ablations(self):
        no_velocity = environment_overrides(self.parser.parse_args(["--no-vel"]))
        no_angular_rate = environment_overrides(self.parser.parse_args(["--no-ang-vel"]))
        low_observation = environment_overrides(self.parser.parse_args(["--low-obs"]))
        self.assertIs(no_velocity["no_vel"], True)
        self.assertIs(no_angular_rate["no_ang_vel"], True)
        self.assertIs(low_observation["low_obs"], True)

    def test_partial_observation_flags_are_omitted_by_default(self):
        overrides = environment_overrides(self.parser.parse_args([]))
        self.assertNotIn("no_vel", overrides)
        self.assertNotIn("no_ang_vel", overrides)
        self.assertNotIn("low_obs", overrides)

    def test_full_observation_flags_disable_ablations(self):
        overrides = environment_overrides(
            self.parser.parse_args(["--with-vel", "--with-ang-vel", "--full-obs"])
        )
        self.assertIs(overrides["no_vel"], False)
        self.assertIs(overrides["no_ang_vel"], False)
        self.assertIs(overrides["low_obs"], False)

    def test_last_observation_flag_wins_for_shared_cli_arrays(self):
        overrides = environment_overrides(
            self.parser.parse_args(["--with-vel", "--no-vel"])
        )
        self.assertIs(overrides["no_vel"], True)


class StatisticsTests(unittest.TestCase):
    def test_ci_and_seed_sampling_unit(self):
        interval = mean_confidence_interval([1.0, 2.0, 3.0])
        self.assertEqual(interval["n"], 3)
        self.assertAlmostEqual(interval["mean"], 2.0)
        self.assertLess(interval["ci_low"], 2.0)
        self.assertGreater(interval["ci_high"], 2.0)
        aggregate = aggregate_training_seeds([{"reward": 1.0}, {"reward": 3.0}], ["reward"])
        self.assertEqual(aggregate["between_training_seeds"]["reward"]["n"], 2)
        self.assertEqual(aggregate["sampling_unit"], "independent_training_seed")

    def test_wilson_bounds(self):
        result = wilson_interval(8, 10)
        self.assertLess(result["ci_low"], 0.8)
        self.assertGreater(result["ci_high"], 0.8)

    def test_success_failure_accounting(self):
        episodes = [
            {"success": True, "failure": False, "termination_reasons": ["success"], "total_reward": 2.0},
            {"success": False, "failure": True, "termination_reasons": ["ground_collision"], "total_reward": -1.0},
            {"success": False, "failure": True, "termination_reasons": ["timeout"], "total_reward": 0.0},
        ]
        summary = aggregate_episodes(episodes)
        self.assertEqual(summary["success_wilson_95"]["successes"], 1)
        self.assertEqual(summary["failure_breakdown"]["ground_collision"]["count"], 1)
        self.assertAlmostEqual(summary["metrics"]["failure"]["mean"], 2.0 / 3.0)


class ProbeTests(unittest.TestCase):
    def test_episode_split_has_no_overlap_and_probe_r2(self):
        episode_ids = np.repeat(np.arange(6), 20)
        train, test = split_episode_ids(episode_ids, seed=4)
        self.assertFalse(set(train) & set(test))
        rng = np.random.default_rng(7)
        hidden = rng.normal(size=(len(episode_ids), 12))
        state = np.zeros((len(episode_ids), 19))
        state[:, :12] = 2.0 * hidden + 0.3
        result = fit_episode_ridge_probe(hidden, state, episode_ids, alpha=0.0, seed=4)
        self.assertTrue(all(value > 0.999999 for value in result["r2"].values()))
        self.assertFalse(set(result["train_episode_ids"]) & set(result["test_episode_ids"]))


class RobustnessTests(unittest.TestCase):
    def test_degradation_direction(self):
        high = degradation(8.0, 10.0, higher_is_better=True)
        low = degradation(1.2, 1.0, higher_is_better=False)
        self.assertAlmostEqual(high["absolute_difference"], -2.0)
        self.assertAlmostEqual(high["percentage_degradation"], 20.0)
        self.assertAlmostEqual(low["percentage_degradation"], 20.0)
        result = robustness_degradation(
            {"success": 1.0, "failure": 0.5}, {"success": 0.5, "failure": 0.75}, ["success", "failure"]
        )
        self.assertAlmostEqual(result["success"]["percentage_degradation"], 50.0)
        self.assertAlmostEqual(result["failure"]["percentage_degradation"], 50.0)


class StabilityUtilityTests(unittest.TestCase):
    def test_central_difference(self):
        matrix = np.asarray([[2.0, -1.0], [0.5, 3.0]])
        jacobian, residual = central_difference_jacobian(lambda value: matrix @ value, np.asarray([0.2, -0.1]), 1e-6)
        np.testing.assert_allclose(jacobian, matrix, atol=1e-9)
        self.assertLess(residual, 1e-8)


if __name__ == "__main__":
    unittest.main()
