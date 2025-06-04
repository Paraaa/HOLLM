import logging
from typing import Dict
from datetime import datetime
import numpy as np
import pandas as pd
from paretoset import paretoset


from mooLLM.benchmarks.benchmark import BENCHMARK
from mooLLM.statistics.statistics import Statistics
from mooLLM.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION
from mooLLM.surrogate_models.surrogate_model import SURROGATE_MODEL
from mooLLM.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER
from mooLLM.warmstarter.warmstarter import WARMSTARTER
from mooLLM.optimization_strategy.optimization_strategy import OptimizationStrategy


logger = logging.getLogger("mooLLM_ToT")


class mooLLMToT(OptimizationStrategy):
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
        self.placebo_interval: int = 0
        self.current_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.tot_settings: Dict = {}

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
        trial = 0
        while trial < self.n_trials:
            depth = self.tot_settings.get("depth", 2)
            logger.debug(f"Starting trial {trial} with depth: {depth}")
            pareto_not_improved = True

            time_taken_candidate_sampler = 0
            time_taken_surrogate_model = 0

            while pareto_not_improved:
                logger.debug(f"Trial: {trial}")

                # get the true pareto set
                true_pareto, true_pareto_fvals = self.statistics.get_pareto_set()
                # initialize the best pareto with the true pareto
                best_pareto = true_pareto
                # true_pareto_fvals = [
                #     {"F1": d["F1"], "F2": d["F2"]} for d in true_pareto_fvals
                # ]
                best_pareto_fvals = true_pareto_fvals

                pareto_fvals_history = []
                pareto_configs_history = []
                pareto_fvals_history.extend(best_pareto_fvals)
                pareto_configs_history.extend(best_pareto)
                logger.debug(
                    f"initial pareto configs history len {len(pareto_configs_history)}"
                )
                for d in range(depth):
                    # get the candidate nodes
                    # provide the path of pareto sets as context in the prompt (true pareto + all pareto sets along the search)
                    (
                        candidate_points,
                        time_taken_candidate_sampler_per_depth,
                        error_rate_candidate_sampler,
                    ) = self.candidate_sampler.get_candidate_points(
                        pareto_front=pareto_configs_history
                    )
                    time_taken_candidate_sampler += (
                        time_taken_candidate_sampler_per_depth
                    )
                    (
                        candidate_evaluations,
                        time_taken_surrogate_model_per_depth,
                        error_rate_surrogate_model,
                    ) = self.surrogate_model.evaluate_candidates(candidate_points)
                    time_taken_surrogate_model += time_taken_surrogate_model_per_depth

                    # group evaluations per node
                    nodes = []
                    idx = 0
                    for node in range(len(candidate_points)):
                        num_candidates_per_node = len(candidate_points[node])
                        nodes.append(
                            candidate_evaluations[idx : idx + num_candidates_per_node]
                        )
                        idx += num_candidates_per_node

                    # candidate_points -> List[List[dict]]  -> num_nodes x candidates per node (can be heterogeneous shape)
                    # candidate evaluations -> List[dict] -> predicted metrics per candidate
                    (
                        best_candidate_index,
                        best_candidate_average_evaluation,
                        hypervolume_contributions,
                    ) = self.acquisition_function.select_candidate_point(
                        true_pareto_fvals, nodes
                    )

                    print(hypervolume_contributions)

                    # pair candidates with their evaluations
                    best_pareto_fvals = nodes[best_candidate_index]
                    best_pareto = candidate_points[best_candidate_index]

                    # only extend the history if the hypervolume contribution is larger than 0?

                    pareto_not_improved = False
                    pareto_fvals_history.extend(best_pareto_fvals)
                    pareto_configs_history.extend(best_pareto)
                    logger.debug(
                        f"pareto configs history len {len(pareto_configs_history)}"
                    )

            filtered_configs = []
            filtered_values = []
            observed_configs, observed_fvals = self.statistics.get_statistics_for_icl()
            for cfg, val in zip(pareto_configs_history, pareto_fvals_history):
                if cfg not in observed_configs:
                    filtered_configs.append(cfg)
                    filtered_values.append(val)

            filtered_values = [[x] for x in filtered_values]

            # No need to select one point, but either take the pareto set of the history and evaluate everything that is not in the history
            # or evaluate everything that improves

            # the history should keep all of the chosen pareto sets
            (
                best_candidate_index,
                best_candidate_average_evaluation,
                hypervolume_contributions,
            ) = self.acquisition_function.select_candidate_point(
                true_pareto_fvals, filtered_values
            )
            print(hypervolume_contributions)

            # get the candidates that have positive hypervolume contribution
            final_configs = [
                filtered_configs[idx]
                for idx, hv_contribution in enumerate(hypervolume_contributions)
                if hv_contribution > 0
            ]
            final_values = [
                filtered_values[idx][0]
                for idx, hv_contribution in enumerate(hypervolume_contributions)
                if hv_contribution > 0
            ]

            # additionally take only the pareto set of the final configs
            print(final_values)
            print(len(final_configs))
            if len(final_configs) > 0:
                pareto_mask = paretoset(
                    pd.DataFrame(final_values), sense=["min", "min"]
                )
                final_configs = list(np.array(final_configs)[pareto_mask])
                print(len(final_configs))

            error_rate = {
                "candidate_sampler": error_rate_candidate_sampler,
                "surrogate_model": error_rate_surrogate_model,
            }

            for best_candidate in final_configs:
                sel_candidate_point, sel_candidate_eval = self.benchmark.evaluate_point(
                    best_candidate
                )
                logger.debug(f"Selected point: {sel_candidate_point}")
                surrogate_model_accuracy = {
                    "surrogate_model_prediction": best_candidate_average_evaluation,
                    "benchmark_evaluation": sel_candidate_eval,
                }
                self.statistics.update(
                    new_config=sel_candidate_point,
                    new_fval=sel_candidate_eval,
                    error_rate=error_rate,
                    surrogate_model_accuracy=surrogate_model_accuracy,
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
                self.statistics.update_tot_data(
                    trial_number=trial,
                    new_config=sel_candidate_point,
                    new_fval=sel_candidate_eval,
                )

            time_taken = {
                "candidate_sampler": time_taken_candidate_sampler,
                "surrogate_model": time_taken_surrogate_model,
                "trial_total_time": (
                    time_taken_candidate_sampler + time_taken_surrogate_model
                ),
            }
            trial += len(final_configs)
            self.statistics.update_time_taken(time_taken)
            self.benchmark.save_progress(self.statistics.get_statistics())

        return self.statistics.get_statistics()
