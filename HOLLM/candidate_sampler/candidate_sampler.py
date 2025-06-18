import time
import json
import logging
from abc import ABC, abstractmethod
from HOLLM.llm.llm import LLMInterface
from HOLLM.statistics.statistics import Statistics
from typing import Dict, List, Tuple
from HOLLM.utils.prompt_builder import PromptBuilder
from HOLLM.benchmarks.benchmark import BENCHMARK


logger = logging.getLogger("CANDIDATE_SAMPLER")


class CANDIDATE_SAMPLER(ABC):
    def __init__(self):
        """
        Initializes the Candidate Sampler class with default values for its components.

        Attributes:
            model (LLMInterface): The language model interface object.
            statistics (Statistics): The statistics object to track observed configurations and evaluations.
            prompt_builder (PromptBuilder): The prompt builder object.
            max_candidates_per_trail (int): The number of candidate points to generate in each iteration.
            candidates_per_request (int): The number of candidate points to generate in each request.
            alpha (float): The alpha value used to calculate the target/desired values.
            context (str): The context scope for generating prompts.
        """
        self.model: LLMInterface = None
        self.statistics: Statistics = None
        self.prompt_builder: PromptBuilder = None
        self.benchmark: BENCHMARK = None
        self.candidates_per_request = None
        self.max_candidates_per_trial = None
        self.alpha = None
        self.metrics = None
        self.tot_settings: Dict = {}
        self.range_parameter_keys: List[str] = None

    @abstractmethod
    def evaluate_desired_values(self, args, **kwargs) -> Dict:
        """
        Evaluates the desired values based on the task context and observed configurations.

        The desired values are the target values for the candidate sampler.
        The method should return a dictionary with the target values.

        Returns:
            Dict: A dictionary with the target values.
        """
        pass

    @abstractmethod
    def generate_points(self, target_number_of_candidates, optionals={}) -> List[Dict]:
        """
        Generates candidate points using the LLM model.

        The method should return a list of dictionaries containing the candidate points of size target_number_of_candidates.

        Args:
            target_number_of_candidates (int): The number of candidate points to generate.

        Returns:
            List[Dict[Any]]: A list of dictionaries containing the candidate points of size target_number_of_candidates or less.
        """
        pass

    def get_candidate_points(
        self, regions_constrains: Dict = {}, optionals={}
    ) -> Tuple[List[Dict], float, float]:
        """
        Generates candidate points using the LLM model.

        The method generates candidate points by repeatedly generating prompts and using the LLM model to generate candidate points.
        The method checks if the suggested points are already in the observed evaluations and filters them out.

        The method returns a list of dictionaries containing the candidate points of size max_candidates_per_trial,
        the time taken to generate the candidate points in seconds,
        and the error rate of the LLM model (i.e. the number of failed LLM calls divided by the total number of calls).

        Returns:
            List[Dict]: A list of dictionaries containing the candidate points of size max_candidates_per_trial.
            float: The time taken to generate the candidate points in seconds.
            float: The error rate of the LLM model (i.e. the number of failed LLM calls divided by the total number of calls).
        """
        start_time = time.time()
        logger.debug("Generating candidate points")

        number_candidate_points = 0
        llm_error_count = 0
        total_llm_calls = 0
        filtered_candidate_points = []

        critical_failure = False
        while number_candidate_points < self.max_candidates_per_trial:
            # try:
            total_llm_calls += 1

            if llm_error_count > 100:
                critical_failure = True
                break

            if llm_error_count > 30:
                target_number_of_candidates = self.max_candidates_per_trial
            else:
                target_number_of_candidates = min(
                    self.max_candidates_per_trial - number_candidate_points,
                    self.candidates_per_request,
                )

            candidate_points = self.generate_points(
                target_number_of_candidates, optionals
            )
            proposed_points = self._filter_candidate_points(
                candidate_points, filtered_candidate_points, regions_constrains
            )

            filtered_candidate_points.extend(proposed_points)
            number_candidate_points = len(filtered_candidate_points)
            logger.debug(f"Number of candidate points: {number_candidate_points}")

            if len(proposed_points) < target_number_of_candidates:
                llm_error_count += 1

            # except Exception as e:
            # logger.warning(f"Failed to generate candidate. Retrying... Error: {e}")

        if critical_failure:
            raise ValueError("too many retries")

        filtered_candidate_points = filtered_candidate_points[
            : self.max_candidates_per_trial
        ]
        end_time = time.time()
        time_taken = end_time - start_time

        if total_llm_calls == 0:
            llm_error_rate = 0
        else:
            llm_error_rate = llm_error_count / total_llm_calls
        return filtered_candidate_points, time_taken, llm_error_rate

    def _filter_candidate_points(
        self, candidate_points, suggested_points, regions_constrains: Dict = {}
    ) -> List[Dict]:
        """
        Filters the proposed samples to ensure they are valid and not duplicates of suggested samples.
        Args:
            candidate_points (list): List of candidate points
            suggested_samples (list): The samples that have already been suggested and should not be duplicated.


        Returns:
            list: List of filtered candidate points

        """
        if isinstance(candidate_points, dict):
            candidate_points = [candidate_points]

        observed_points = [
            json.dumps(observed_point)
            for observed_point in self.statistics.observed_configs
        ]
        # This list(set(...)) is to remove duplicates
        new_candidate_points = list(
            set([json.dumps(candidate_point) for candidate_point in candidate_points])
        )

        # Filter out candidates that already exist in observed points and are valid points
        filtered_candidates = [
            json.loads(candidate_point)
            for candidate_point in new_candidate_points
            if candidate_point not in observed_points
            and self.benchmark.is_valid_candidate(json.loads(candidate_point))
            and json.loads(candidate_point) not in suggested_points
        ]

        if regions_constrains:
            logger.debug("Filtering candidates based on region constraints")
            logger.debug(f"Region constraints: {regions_constrains}")
            logger.debug(
                f"Filtered candidates before region constraints: {filtered_candidates}"
            )

            filtered_candidates = [
                candidate
                for candidate in filtered_candidates
                # This is the case for choice type features
                if all(
                    candidate.get(feature) is not None
                    and candidate.get(feature) in allowed_values
                    for feature, allowed_values in regions_constrains.items()
                    if feature not in self.range_parameter_keys
                )
                # This is the case for range type features
                and all(
                    candidate.get(feature) is not None
                    and candidate.get(feature) <= allowed_values[1] + 1e6
                    and candidate.get(feature) >= allowed_values[0] - 1e6
                    for feature, allowed_values in regions_constrains.items()
                    if feature in self.range_parameter_keys
                )
            ]

            logger.debug(
                f"Filtered candidates after region constraints: {filtered_candidates}"
            )

        return filtered_candidates
