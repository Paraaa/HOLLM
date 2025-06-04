import time
import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from mooLLM.benchmarks.benchmark import BENCHMARK
from mooLLM.statistics.statistics import Statistics
from mooLLM.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION
from mooLLM.surrogate_models.surrogate_model import SURROGATE_MODEL
from mooLLM.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER
from mooLLM.warmstarter.warmstarter import WARMSTARTER
from mooLLM.optimization_strategy.optimization_strategy import OptimizationStrategy
from mooLLM.space_partitioning.utils import Region
from mooLLM.space_partitioning.space_partitioning_strategy import (
    SPACE_PARTITIONING_STRATEGY,
)
from mooLLM.region_acquisition_functions.region_acq import RegionACQ
from mooLLM.optimization_strategy.threaded_mooLLM import ThreadedMOOLLM

logger = logging.getLogger("SpacePartitioningMOOLLM")


class SpacePartitioningMOOLLM(OptimizationStrategy):
    def __init__(self):
        self.benchmark: BENCHMARK = None
        self.search_space_partitioning: SPACE_PARTITIONING_STRATEGY = None
        self.statistics: Statistics = None
        self.warmstarter: WARMSTARTER = None
        self.n_trials = None
        self.acquisition_function: ACQUISITION_FUNCTION = None
        self.region_acquisition_function: RegionACQ = None

        # threaded instance specific attributes
        self.surrogate_model: SURROGATE_MODEL = None
        self.candidate_sampler: CANDIDATE_SAMPLER = None

        # Defines the upper limit on the number of partitions to be created per trial
        self.partitions_per_trial: int = None

        # Defines the number of points to be selected from the set of candidate points after a trial to be evaluated
        self.top_k: int = None

        # Defines whether to use clustering or not
        self.use_clustering: bool = False

        self.use_pareto_front_as_regions: bool = False

    def initialize(self):
        init_configs = self.warmstarter.generate_initialization()

        for config in init_configs:
            cfg, cfg_fvals = self.benchmark.evaluate_point(config)
            self.statistics.observed_configs.append(cfg)
            self.statistics.observed_fvals.append(cfg_fvals)

        logger.debug(
            f"Initialized mooLLM with configs: {self.statistics.observed_configs}"
        )
        logger.debug(f"Initialized mooLLM with fvals: {self.statistics.observed_fvals}")

    def optimize(self):
        top_k_candidates, statistics = self.optimize_threaded()
        return top_k_candidates, statistics

    def _generate_regions(self):
        if self.use_pareto_front_as_regions:
            logger.debug("Using Pareto front as regions")
            configs, evaluations = self.statistics.get_pareto_set()
        else:
            logger.debug("Using ICL statistics for regions")
            configs, evaluations = self.statistics.get_statistics_for_icl()
        regions = self.search_space_partitioning.partition(configs, evaluations)
        return regions

    def optimize_threaded(self):
        for trial in range(self.n_trials):
            logger.info(f"Starting trial: {trial}")
            start_time = time.time()

            all_regions = self._generate_regions()
            logger.debug(f"Number of regions: {len(all_regions)}")

            regions = self.region_acquisition_function.select_regions(
                all_regions, self.partitions_per_trial
            )

            logger.debug(f"Selected regions: {regions}")

            # icl_clusters = self.create_icl_clusters(regions) # Clusters by configuration
            icl_clusters = self.create_icl_clusters_by_fvals(
                regions
            )  # Clusters by function values

            logger.debug(f"Clustered history: {icl_clusters}")
            instances: List[ThreadedMOOLLM] = self.create_instances(
                regions, icl_clusters
            )
            logger.debug(f"Created threaded instances: {instances}")
            tasks = [instance.optimize_threaded() for instance in instances]
            logger.debug(f"Starting threaded optimization tasks: {tasks}")

            results = []
            with ThreadPoolExecutor(max_workers=self.partitions_per_trial) as executor:
                # Submit all tasks to the executor
                future_to_instance = {
                    executor.submit(instance.optimize_threaded): instance
                    for instance in instances
                }

                # Collect results as they complete
                for future in as_completed(future_to_instance):
                    instance = future_to_instance[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.debug(
                            f"Completed optimization for region: {instance.region}"
                        )
                    except Exception as exc:
                        logger.error(
                            f"Instance {instance.instance_index} generated an exception: {exc}"
                        )

            logger.info(f"All threaded optimization tasks have completed: {results}")
            candidate_configs, candidate_evaluations = self.reorganize_data(results)

            (
                best_candidate_indices,
                best_candidate_evaluations,
                _,
            ) = self.acquisition_function.select_candidate_point(
                candidate_evaluations, top_k=self.top_k
            )

            top_k_candidates = [candidate_configs[i] for i in best_candidate_indices]

            for candidate in top_k_candidates:
                sel_candidate_point, sel_candidate_eval = self.benchmark.evaluate_point(
                    candidate
                )
                if sel_candidate_point and sel_candidate_eval:
                    self.update_statistics(sel_candidate_point, sel_candidate_eval)
                else:
                    logger.debug(
                        f"No evaluation for {candidate} provided from the benchmark. Skipping statistics update."
                    )

            end_time = time.time()
            time_taken = {
                "trial_total_time": (end_time - start_time),
            }
            self.statistics.update_time_taken(time_taken)
            # self.benchmark.save_progress(self.statistics.get_statistics())

        return top_k_candidates, self.statistics.get_statistics()

    def _get_top_k_indices(self, results, k):
        np_results = np.array(results)
        sorted_indices = np.argsort(np_results)[::-1]
        return sorted_indices[:k]

    def create_instances(
        self, regions: List[Region], icl_clusters: List
    ) -> List[ThreadedMOOLLM]:
        threaded_instances = []
        for index, (region, icl_cluster) in enumerate(zip(regions, icl_clusters)):
            threaded_moo_llm = ThreadedMOOLLM()
            threaded_moo_llm.acquisition_function = self.acquisition_function
            threaded_moo_llm.surrogate_model = self.surrogate_model
            threaded_moo_llm.candidate_sampler = self.candidate_sampler
            threaded_moo_llm.statistics = self.statistics
            threaded_moo_llm.region = region
            if self.use_clustering:
                threaded_moo_llm.region_icl_examples = icl_cluster
            else:
                threaded_moo_llm.region_icl_examples = (
                    self.statistics.get_statistics_for_icl()
                )
            threaded_moo_llm.instance_index = index
            threaded_instances.append(threaded_moo_llm)
        return threaded_instances

    def reorganize_data(self, data):
        all_configs = []
        all_evaluations = []

        for configs, evaluations in data:
            all_configs.extend(configs)
            all_evaluations.extend(evaluations)

        return all_configs, all_evaluations

    def create_icl_clusters(
        self, regions: List[Region]
    ) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        Clusters historical points into the provided regions. Each region has a
        center (as a dictionary of coordinate values) that will be used as the
        reference point for clustering the historical points.

        Assumptions:
        - The history of points is available through self.statistics.observed_configs,
            where each point is a dictionary with the same keys as region.center.

        Parameters:
            regions (List[Region]): List of Region objects that specify the regions
                                    with a center for clustering.

        Returns:
            List[Tuple[List[Dict], List[Dict]]]: A list where each element is a tuple of two lists:
                            the first list contains points (dictionaries) that belong to the respective region,
                            and the second list contains the corresponding function values.
        """

        # Retrieve the historical points and their function values.
        history_configs, history_fvals = self.statistics.get_statistics_for_icl()

        # We'll assume that all points and region centers share the same coordinate keys.
        # For instance, if your points are like {"x": 1.0, "y": 2.0}, region.center should be similar.
        coordinate_keys = list(regions[0].center.keys())

        # Initialize the clusters: create one empty list for each region.
        clusters = [(list(), list()) for _ in regions]

        # Iterate over each historical point and its corresponding function value.
        for config, fval in zip(history_configs, history_fvals):
            # Convert the point into a numpy array in the order of coordinate_keys.
            point_coords = np.array([config[dim] for dim in coordinate_keys])

            # Compute distances from this point to each region center.
            distances = []
            for region in regions:
                # Get the center coordinates (as an array) for the current region.
                center_coords = np.array(
                    [region.center[dim] for dim in coordinate_keys]
                )
                # Compute the Euclidean distance.
                distance = np.linalg.norm(point_coords - center_coords)
                distances.append(distance)

            # Find the region index with the minimum distance.
            nearest_region_idx = int(np.argmin(distances))
            # Append the current point and its function value to the list corresponding to that region.
            clusters[nearest_region_idx][0].append(config)
            clusters[nearest_region_idx][1].append(fval)

        return clusters

    def create_icl_clusters_by_fvals(
        self, regions: List[Region]
    ) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        Clusters historical configurations and their function evaluations into the provided regions,
        but using the function values (fvals) as the basis for clustering rather than the original
        configuration features. That is, each historical point is assigned to the region whose fval
        reference (assumed to be stored in region.center) is closest in Euclidean space.

        Assumptions:
        - The history of points and their function values are available via:
                history_configs, history_fvals = self.statistics.get_statistics_for_icl()
        - Each fval is a dictionary with numeric values (e.g. {"f1": ..., "f2": ...}),
            and all fvals share the same keys.
        - The provided regions are assumed to contain reference fval values in their 'center' attribute.
            (If that is not the case, consider adding a separate attribute, e.g. 'fval_center',
            and modifying the references below.)

        Parameters:
            regions (List[Region]): List of Region objects that specify the regions with a fval
                                    reference point for clustering.

        Returns:
            List[Tuple[List[Dict], List[Dict]]]: A list where each element is a tuple containing two lists:
                - The first list contains configuration dictionaries that belong to this region.
                - The second list contains the corresponding function evaluation dictionaries.
        """
        history_configs, history_fvals = self.statistics.get_statistics_for_icl()

        if not history_fvals:
            return [(list(), list()) for _ in regions]

        fval_keys = list(regions[0].center_fval.keys())

        clusters = [(list(), list()) for _ in regions]

        for config, fval in zip(history_configs, history_fvals):
            fval_vector = np.array([fval[key] for key in fval_keys])

            distances = []
            for region in regions:
                region_fval_vector = np.array(
                    [region.center_fval[key] for key in fval_keys]
                )
                distance = np.linalg.norm(fval_vector - region_fval_vector)
                distances.append(distance)

            nearest_region_idx = int(np.argmin(distances))

            clusters[nearest_region_idx][0].append(config)
            clusters[nearest_region_idx][1].append(fval)

        return clusters

    def update_statistics(self, sel_candidate_point, sel_candidate_eval):
        # Note: This is primarily needed for the syne-tune implementation as we the the results from the framework and not
        # from out benchmark interface. We have to be careful to not save the same point multiple times/twice.
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
