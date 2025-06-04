import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from mooLLM.statistics.context_limit_strategy.context_limit_strategy import (
    ContextLimitStrategy,
)
from paretoset import paretoset

logger = logging.getLogger("STATISTICS")


class Statistics:
    def __init__(self):
        self.observed_configs = []
        self.observed_fvals = []
        self.observed_configs_per_tot_trial = []
        self.observed_fvals_per_tot_trial = []
        self.error_rate_per_trial = []
        self.time_taken_per_trial = []
        self.surrogate_model_accuracy_per_trial = []
        self.cost_per_request = []
        self.token_usage_per_request = []

        self.all_configs_per_trial = []

        self.all_hv_contributions_per_trial = []

        # This is use if we hit the context limit
        self.context_configs = []
        self.context_fvals = []

        self.seed = None
        self.initial_samples = None
        self.benchmark_name: str = None
        self.model_name = None

        self.metrics: List[str] = None

        self.max_context_configs = None
        self.context_limit_strategy: ContextLimitStrategy = None
        self.total_time_taken = 0

    def update(
        self,
        error_rate: Dict,
        all_configs_per_trial: Dict,
        all_hv_contributions_per_trial: Dict,
    ) -> None:
        """
        Updates the statistics with new data for one trial at a time.

        Args:
            new_configs (Dict): New configurations to add to the observed configs.
            new_fvals (Dict): New function values to add to the observed fvals.
            error_rate (Dict): Error rate to add to the error rate per trial.
            surrogate_model_accuracy (Dict): Accuracy of the surrogate model to add to the surrogate model accuracy per trial
        """
        self.all_configs_per_trial.append(all_configs_per_trial)
        self.all_hv_contributions_per_trial.append(all_hv_contributions_per_trial)

    def update_fvals(
        self,
        new_config: Dict,
        new_fval: Dict,
        surrogate_model_accuracy: Dict,
    ) -> None:
        """
        Updates the statistics with new data for one trial at a time.

        Args:
            new_configs (Dict): New configurations to add to the observed configs.
            new_fvals (Dict): New function values to add to the observed fvals.
            error_rate (Dict): Error rate to add to the error rate per trial.
            surrogate_model_accuracy (Dict): Accuracy of the surrogate model to add to the surrogate model accuracy per trial
        """
        self.observed_configs.append(new_config)
        self.observed_fvals.append(new_fval)
        self.surrogate_model_accuracy_per_trial.append(surrogate_model_accuracy)

        logger.debug(f"Updated mooLLM with configs: {self.observed_configs}")
        logger.debug(f"Updated mooLLM with fvals: {self.observed_fvals}")

        if len(self.observed_configs) > self.max_context_configs:
            logger.debug(
                f"Hitting defined context limit: {self.max_context_configs}. Using {self.context_limit_strategy.__class__.__name__} strategy."
            )
            self.context_configs, self.context_fvals = (
                self.context_limit_strategy.update_context(
                    self.observed_configs, self.observed_fvals
                )
            )
        else:
            self.context_configs.append(new_config)
            self.context_fvals.append(new_fval)

    def update_tot_data(
        self,
        trial_number: int,
        new_config: Dict,
        new_fval: Dict,
    ) -> None:
        """
        Updates the statistics with new data for the total trial.

        Args:
            trial_number (int): The total trial number.
            new_config (Dict): New configurations to add to the observed configs per trial.
            new_fval (Dict): New function values to add to the observed fvals per trial.
        """
        self.observed_configs_per_tot_trial.append(
            {
                "config": new_config,
                "trial": trial_number,
            }
        )
        self.observed_fvals_per_tot_trial.append(
            {
                "fval": new_fval,
                "trial": trial_number,
            }
        )

    def update_time_taken(self, time_taken: Dict):
        """
        Updates the time taken statistics for one trial at a time.

        Args:
            time_taken (Dict): Time taken for one trial.
        """
        self.time_taken_per_trial.append(time_taken)
        self.total_time_taken += time_taken.get("trial_total_time", 0)
        logger.debug(
            f"Time taken for trial: {time_taken.get('trial_total_time', 0)} seconds"
        )

    def update_cost(self, cost: Dict) -> None:
        """
        Updates the cost of the request.

        Args:
            cost (Dict): Dict: A dictionary containing prompt_cost, completion_cost, and total_cost in USD.
        """
        self.cost_per_request.append(cost)
        logger.info(f"Cost of the request: {cost} USD")

    def update_token_usage(self, usage: Dict) -> None:
        """
        Updates the token usage of the request.

        Args:
            usage (Dict): Dict: A dictionary containing the token usage of the request.
        """
        self.token_usage_per_request.append(
            {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        )
        logger.info(f"Token usage of the request: {usage}")

    def get_statistics_for_icl(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Returns the statistics for the ICL examples.

        If the number of observed configurations exceeds the maximum context limit,
        the context configurations and function values are returned. Otherwise, the
        full observed configurations and function values are returned directly.

        Returns:
            Tuple[List[Dict], List[Dict]]: The ICL examples.
        """
        if len(self.observed_configs) > self.max_context_configs:
            return self.context_configs, self.context_fvals
        return self.observed_configs, self.observed_fvals

    def get_pareto_set(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Returns the pareto set.

        Returns:
            Tuple[List[Dict], List[Dict]]: The pareto set configurations and function values.
        """
        observed_fvals_df = pd.DataFrame(self.observed_fvals)
        pareto_mask = paretoset(observed_fvals_df, sense=["min", "min"])

        pareto_configs = list(np.array(self.observed_configs)[pareto_mask])
        pareto_fvals = list(np.array(self.observed_fvals)[pareto_mask])
        return pareto_configs, pareto_fvals

    def get_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Get all statistics as pandas DataFrames.

        Returns:
            Dict: {
                "observed_configs": pd.DataFrame,
                "observed_fvals": pd.DataFrame,
                "error_rate_per_trials": pd.DataFrame,
                "time_taken_per_trials": pd.DataFrame
                "surrogate_model_accuracy_per_trial": pd.DataFrame
                "cost_per_request": pd.DataFrame
                "token_usage_per_request": pd.DataFrame
                "all_configs_per_trial": pd.DataFrame
                "all_hv_contributions_per_trial": pd.DataFrame
                "observed_configs_per_tot_trial": pd.DataFrame
                "observed_fvals_per_tot_trial": pd.DataFrame
            }
        """
        return {
            "observed_configs": pd.DataFrame(self.observed_configs),
            "observed_fvals": pd.DataFrame(self.observed_fvals),
            "error_rate_per_trials": pd.DataFrame(self.error_rate_per_trial),
            "time_taken_per_trials": pd.DataFrame(self.time_taken_per_trial),
            "surrogate_model_accuracy_per_trial": pd.DataFrame(
                self.surrogate_model_accuracy_per_trial
            ),
            "cost_per_request": pd.DataFrame(self.cost_per_request),
            "token_usage_per_request": pd.DataFrame(self.token_usage_per_request),
            "all_configs_per_trial": pd.DataFrame(self.all_configs_per_trial),
            "all_hv_contributions_per_trial": pd.DataFrame(
                self.all_hv_contributions_per_trial
            ),
            "observed_configs_per_tot_trial": pd.DataFrame(
                self.observed_configs_per_tot_trial
            ),
            "observed_fvals_per_tot_trial": pd.DataFrame(
                self.observed_fvals_per_tot_trial
            ),
        }
