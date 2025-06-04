import logging
import os
import pandas as pd
from typing import List, Dict
from hwgpt.api import HWGPT
from mooLLM.benchmarks.benchmark import BENCHMARK

logger = logging.getLogger("HW_GPT_BENCH")


class HWGPTBENCH(BENCHMARK):
    def __init__(
        self,
        scale: str = "s",
        use_supernet_surrogate: bool = False,
        metrics: List[str] = ["perplexity", "latency", "energy"],
        device: str = "rtx2080",
        predictor: str = "mlp",
        model_name: str = None,
        seed: int = None,
    ):
        self.api = HWGPT(
            search_space=scale,
            use_supernet_surrogate=use_supernet_surrogate,
            base_path="../HW-GPT-Bench",
        )
        self.device = device
        self.metrics = metrics
        self.predictor = predictor
        self.scale = scale
        self.use_supernet_surrogate = use_supernet_surrogate
        self.benchmark_name = "HWGPT"
        self.model_name = model_name
        self.seed = seed

    def generate_initialization(self, n_points: int, **kwargs):
        init_configs = []
        for _ in range(n_points):
            random_arch = self.api.sample_arch()
            init_configs.append(random_arch)
        return init_configs

    def evaluate_point(self, point, **kwargs):
        evaluation = {}
        point.update({"sample_layer_indices": list(range(point.get("sample_n_layer")))})
        point.update({"sample_bias": f"{point.get('sample_bias') == 'true'}"})
        self.api.set_arch(point)
        for metric in self.metrics:
            if metric != "Perplexity":
                if metric == "Latency":
                    evaluation_metric = "latencies"
                elif metric == "Energy":
                    evaluation_metric = "energies"
                evaluation[metric] = (
                    self.api.query(
                        metric=evaluation_metric,
                        predictor=self.predictor,
                        device=self.device,
                    )
                    .get(evaluation_metric)
                    .get(self.device)
                )
            else:
                evaluation_metric = "perplexity"
                evaluation[metric] = self.api.query(
                    metric=evaluation_metric,
                    predictor=self.predictor,
                    device=self.device,
                ).get(evaluation_metric)
        point.update({"sample_bias": str(point.get("sample_bias")).lower()})
        return point, evaluation

    def get_few_shot_samples(self, **kwargs):
        # Using the heuristic that the largest and smallest model show the range of the metrics
        model_config = self.api.cfg_model.model
        largest_model = {
            "sample_embed_dim": sorted(model_config.embed_choices)[-1],
            "sample_n_layer": sorted(model_config.layer_choices)[-1],
            # "sample_layer_indices": list(range(sorted(model_config.layer_choices)[-1])),
            "sample_n_head": [sorted(model_config.head_choices)[-1]]
            * sorted(model_config.layer_choices)[-1],
            "sample_mlp_ratio": [sorted(model_config.mlp_ratio_choices)[-1]]
            * sorted(model_config.layer_choices)[-1],
            "sample_bias": "true",
        }

        # Middle Model
        middle_model = {
            "sample_embed_dim": sorted(model_config.embed_choices)[1],
            "sample_n_layer": sorted(model_config.layer_choices)[1],
            # "sample_layer_indices": list(range(sorted(model_config.layer_choices)[1])),
            "sample_n_head": [sorted(model_config.head_choices)[1]]
            * sorted(model_config.layer_choices)[1],
            "sample_mlp_ratio": [sorted(model_config.mlp_ratio_choices)[1]]
            * sorted(model_config.layer_choices)[1],
            "sample_bias": "false",
        }

        # Smallest Model
        smallest_model = {
            "sample_embed_dim": sorted(model_config.embed_choices)[0],
            "sample_n_layer": sorted(model_config.layer_choices)[0],
            # "sample_layer_indices": list(range(sorted(model_config.layer_choices)[0])),
            "sample_n_head": [sorted(model_config.head_choices)[0]]
            * sorted(model_config.layer_choices)[0],
            "sample_mlp_ratio": [sorted(model_config.mlp_ratio_choices)[0]]
            * sorted(model_config.layer_choices)[0],
            "sample_bias": "false",
        }

        large_model_evaluations = self.evaluate_point(largest_model)[1]
        middle_model_evaluations = self.evaluate_point(middle_model)[1]
        small_model_evaluations = self.evaluate_point(smallest_model)[1]
        few_show_examples = [
            (largest_model, large_model_evaluations),
            (middle_model, middle_model_evaluations),
            (smallest_model, small_model_evaluations),
        ]
        return few_show_examples

    def get_metrics_ranges(self, **kwargs):
        few_shot_examples = self.get_few_shot_samples()
        evaluations = pd.DataFrame([evaluation for _, evaluation in few_shot_examples])
        metrics_ranges = {
            metric: [evaluations[metric].min(), evaluations[metric].max()]
            for metric in evaluations.columns
        }
        return metrics_ranges

    def is_valid_candidate(self, candidate):
        try:
            # all keys are present
            are_keys_present = (
                "sample_embed_dim" in candidate.keys()
                and "sample_n_layer" in candidate.keys()
                and "sample_n_head" in candidate.keys()
                and "sample_mlp_ratio" in candidate.keys()
                # and "sample_layer_indices" in candidate.keys()
                and "sample_bias" in candidate.keys()
            )
            if not are_keys_present:
                return False

            # their values are proper types
            # their values are in their ranges/discrete values
            # sample_n_head and mlp ratio are of the proper size (at least sample_n_layer)

            is_embed_dim_valid = (
                candidate["sample_embed_dim"] in self.api.cfg_model.model.embed_choices
            )

            is_num_layers_valid = (
                candidate["sample_n_layer"] in self.api.cfg_model.model.layer_choices
            )

            # is_layer_indices_valid_size = (
            #     len(candidate["sample_layer_indices"]) == candidate["sample_n_layer"]
            # )

            is_num_heads_valid_size = (
                len(candidate["sample_n_head"]) >= candidate["sample_n_layer"]
            )
            is_num_heads_valid = all(
                n_head in self.api.cfg_model.model.head_choices
                for n_head in candidate["sample_n_head"]
            )

            is_mlp_ratio_valid_size = (
                len(candidate["sample_mlp_ratio"]) >= candidate["sample_n_layer"]
            )
            is_mlp_ratio_valid = all(
                mlp_ratio in self.api.cfg_model.model.mlp_ratio_choices
                for mlp_ratio in candidate["sample_mlp_ratio"]
            )

            # is_bias_valid = candidate["sample_bias"] in ["true", "false"]

            no_error_present = True
            if not are_keys_present:
                logger.warning("Error: Required keys are missing.")
                no_error_present = False
            if not is_embed_dim_valid:
                logger.warning("Error: The embedding dimension is invalid.")
                no_error_present = False
            if not is_num_layers_valid:
                logger.warning("Error: The number of layers is invalid.")
                no_error_present = False
            # if not is_layer_indices_valid_size:
            #     logger.warning("Error: The layer indices size is invalid.")
            #     no_error_present = False
            if not is_num_heads_valid_size:
                logger.warning("Error: The number of heads size is invalid.")
                no_error_present = False
            if not is_num_heads_valid:
                logger.warning("Error: The number of heads is invalid.")
                no_error_present = False
            if not is_mlp_ratio_valid_size:
                logger.warning("Error: The MLP ratio size is invalid.")
                no_error_present = False
            if not is_mlp_ratio_valid:
                logger.warning("Error: The MLP ratio is invalid.")
                no_error_present = False
            # if not is_bias_valid:
            #     logger.warning("Error: The bias setting is invalid.")
            #     no_error_present = False

            return no_error_present

        except Exception as e:
            logger.warning(e)
            return False

    def is_valid_evaluation(self, evaluation):
        is_numeric_and_positive = all(
            isinstance(value, (int, float)) and value > 0
            for value in evaluation.values()
        )
        return is_numeric_and_positive

    def save_progress(self, statistics: Dict) -> None:
        hw_metric = [metric for metric in self.metrics if metric != "perplexity"][0]

        logger.debug(
            f"Saving progress {statistics}",
        )
        for key, statistic in statistics.items():
            fval_dir = f"./results/{self.benchmark_name}/{self.method_name}/{key}"
            fval_filename = f"{self.model_name}_{hw_metric}_{self.device}_{self.scale}_{self.seed}.csv"
            os.makedirs(fval_dir, exist_ok=True)
            statistic.to_csv(f"{fval_dir}/{fval_filename}", index=False)
            logger.debug(f"Writing {key} to {fval_dir}/{fval_filename}")
