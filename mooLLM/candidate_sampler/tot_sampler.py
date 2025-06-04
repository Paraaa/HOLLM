import logging
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple
from mooLLM.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER
from mooLLM.acquisition_functions.hypervolume_improvement import HypervolumeImprovement
from mooLLM.statistics.statistics import Statistics

logger = logging.getLogger("TREE_OF_THOUGHT_SAMPLER")


class TreeOfThoughtSampler(CANDIDATE_SAMPLER):
    def __init__(self):
        super().__init__()

    def generate_points(self, target_number_of_candidates: int):
        pass

    def get_candidate_points(self, pareto_front=None) -> List[List[Dict]]:
        start_time = time.time()

        logger.debug(
            f"Generating candidate nodes: {self.tot_settings.get('depth', 2)} levels, {self.tot_settings.get('nodes_per_level', 3)} nodes per level, {self.tot_settings.get('candidates_per_node', 3)} candidates per node"
        )
        candidate_nodes_at_current_depth = self._generate_configurations(pareto_front)

        end_time = time.time()
        time_taken = end_time - start_time
        return candidate_nodes_at_current_depth, time_taken, 0

    def _generate_configurations(self, pareto_front) -> List[Dict]:
        while True:
            try:
                prompt = self.prompt_builder.build_prompt(
                    phase="candidate generation",
                    pareto_front=pareto_front,
                    nodes_per_level=self.tot_settings.get("nodes_per_level", 3),
                    candidates_per_node=self.tot_settings.get("candidates_per_node", 3),
                )
                response = self.model.prompt(prompt, max_number_of_tokens=5000)
                logger.debug(f"Prompt generation: {prompt}")
                logger.debug(f"Prompt response: {response}")
                nodes_current_level = self.model.to_json(response)

                # flatten candidate_points for validation
                are_candidates_valid = np.array(
                    [
                        self.benchmark.is_valid_candidate(candidate)
                        for candidates_per_node in nodes_current_level
                        for candidate in candidates_per_node
                    ]
                )

                if np.any(are_candidates_valid == False):  # noqa: E712
                    logger.warning("Not all generated candidates are valid")
                    continue
                return nodes_current_level
            except Exception as e:
                logger.warning(f"Generation failed: {e}")

    def _is_valid_config(self, candidate: Dict) -> bool:
        return self.benchmark.is_valid_candidate(candidate.get("configuration", {}))

    def _is_valid_metrics(self, metrics: Dict) -> bool:
        return self.benchmark.is_valid_evaluation(metrics.get("performance", {}))

    def _average_evaluations(self, evaluations: List[List[Dict]]) -> List[Dict]:
        return [pd.DataFrame(evals).mean().to_dict() for evals in evaluations]

    def evaluate_desired_values(self):
        pass
