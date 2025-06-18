import logging
from typing import Dict
from datetime import datetime


from HOLLM.benchmarks.benchmark import BENCHMARK
from HOLLM.statistics.statistics import Statistics
from HOLLM.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION
from HOLLM.surrogate_models.surrogate_model import SURROGATE_MODEL
from HOLLM.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER
from HOLLM.warmstarter.warmstarter import WARMSTARTER
from HOLLM.optimization_strategy.optimization_strategy import OptimizationStrategy


logger = logging.getLogger("mooLLM")


class mooLLM(OptimizationStrategy):
    def __init__(self):
        """
        Initializes the mooLLM class with default values for its components.

        Attributes:
            warmstarter (WARMSTARTER): The warmstarter object for generating initial samples.
            candidate_sampler (CANDIDATE_SAMPLER): The sampler for generating candidate configuration points.
            acquisition_function (ACQUISITION_FUNCTION): The function used to evaluate candidate points.
            surrogate_model (SURROGATE_MODEL): The model used to predict function values of candidate points.
            statistics (Statistics): The statistics object to track observed configurations and evaluations.
            benchmark (BENCHMARK): The benchmark used for evaluating configuration points.
            initial_samples (int): The number of initial samples to generate and evaluate.
            n_trials (int): The number of optimization trials to perform.
        """
        self.warmstarter: WARMSTARTER = None
        self.candidate_sampler: CANDIDATE_SAMPLER = None
        self.acquisition_function: ACQUISITION_FUNCTION = None
        self.surrogate_model: SURROGATE_MODEL = None
        self.statistics: Statistics = None
        self.benchmark: BENCHMARK = None
        self.initial_samples: int = None
        self.n_trials: int = None
        self.start_from_trial: int = 0
        self.current_trial: int = 0

    def initialize(self):
        """
        Initializes the mooLLM instance by generating initial configurations
        and evaluating them using the benchmark. The evaluated configurations
        and their function values are stored in the statistics object.

        This method uses the warmstarter to generate a set of initial
        configuration points, evaluates each point with the benchmark, and
        records the results. It logs the initialized configurations and their
        corresponding function values.

        Raises:
            Exception: If the warmstarter or benchmark is not set.
        """
        init_configs = self.warmstarter.generate_initialization()

        for config in init_configs:
            cfg, cfg_fvals = self.benchmark.evaluate_point(config)
            self.statistics.observed_configs.append(cfg)
            self.statistics.observed_fvals.append(cfg_fvals)

        logger.debug(
            f"Initialized mooLLM with configs: {self.statistics.observed_configs}"
        )
        logger.debug(f"Initialized mooLLM with fvals: {self.statistics.observed_fvals}")

    def optimize(self) -> Dict:
        """
        Runs the optimization loop of the mooLLM algorithm.

        This method runs the optimization loop, which consists of generating
        candidate points, evaluating them using the surrogate model, selecting
        the best candidate using the acquisition function, evaluating the best
        candidate using the benchmark, and updating the statistics object
        with the new observations.

        It logs the current trial number, the best candidate, the selected point,
        the updated configurations, the updated function values, and the time
        taken for each trial.

        The optimization loop is run for the specified number of n_trials.

        Returns:

        Get all statistics as pandas DataFrames.

            Dict: {
                "observed_configs": pd.DataFrame,
                "observed_fvals": pd.DataFrame,
                "error_rate_per_trials": pd.DataFrame,
                "time_taken_per_trials": pd.DataFrame
                "surrogate_model_accuracy_per_trial": pd.DataFrame
                "cost_per_request": pd.DataFrame
            }
        Raises:
            Exception: If the candidate_sampler, surrogate_model, or benchmark
                is not set.
        """
        for trial in range(self.start_from_trial, self.n_trials):
            self.current_trial = trial
            logger.debug(f"Trial: {self.current_trial}")

            (
                candidate_points,
                time_taken_candidate_sampler,
                error_rate_candidate_sampler,
            ) = self.candidate_sampler.get_candidate_points()

            (
                candidate_evaluations,
                time_taken_surrogate_model,
                error_rate_surrogate_model,
            ) = self.surrogate_model.evaluate_candidates(candidate_points)

            (
                best_candidate_index,
                best_candidate_average_evaluation,
                hypervolume_contributions,
            ) = self.acquisition_function.select_candidate_point(candidate_evaluations)
            logger.debug(
                f"Best candidate index: {best_candidate_index}, Average evaluation: {best_candidate_average_evaluation}"
            )

            best_candidate = candidate_points[best_candidate_index]
            logger.debug(f"Best candidate: {best_candidate}")

            ### Update statistics

            error_rate = {
                "candidate_sampler": error_rate_candidate_sampler,
                "surrogate_model": error_rate_surrogate_model,
            }

            if best_candidate:
                sel_candidate_point, sel_candidate_eval = self.benchmark.evaluate_point(
                    best_candidate
                )
                if sel_candidate_point and sel_candidate_eval:
                    self.update_statistics(sel_candidate_point, sel_candidate_eval)
                else:
                    logger.debug(
                        f"No evaluation for {best_candidate} provided from the benchmark. Skipping statistics update."
                    )

            self.statistics.update(
                error_rate=error_rate,
                all_configs_per_trial={
                    "candidate_points": candidate_points,
                    "best_candidate": best_candidate,
                    "best_candidate_index": best_candidate_index,
                },
                all_hv_contributions_per_trial={
                    "candidate_evaluations": candidate_evaluations,
                    "hypervolume_contributions": hypervolume_contributions,
                },
            )

            time_taken = {
                "candidate_sampler": time_taken_candidate_sampler,
                "surrogate_model": time_taken_surrogate_model,
                "trial_total_time": (
                    time_taken_candidate_sampler + time_taken_surrogate_model
                ),
            }

            self.statistics.update_time_taken(time_taken)

        return best_candidate, self.statistics.get_statistics()

    def update_statistics(self, sel_candidate_point, sel_candidate_eval):
        logger.debug(f"Selected point: {sel_candidate_point}")

        surrogate_model_accuracy = {
            "surrogate_model_prediction": 0,  # TODO #best_candidate_average_evaluation,
            "benchmark_evaluation": sel_candidate_eval,
        }

        self.statistics.update_fvals(
            new_config=sel_candidate_point,
            new_fval=sel_candidate_eval,
            surrogate_model_accuracy=surrogate_model_accuracy,
        )

        self.benchmark.save_progress(self.statistics.get_statistics())
