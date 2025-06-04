import os
import time
import torch
import random
import logging
import pathlib
import pandas as pd
from lm_eval import evaluator
from typing import List, Tuple, Dict, Any
from hwgpt.api import HWGPT
from mooLLM.benchmarks.benchmark import BENCHMARK
from litgpt import Config
from litgpt.scripts.download import download_from_hub
from whittle.models.gpt.model import GPT
import torch
import transformers
from whittle.tutorials.gpt_utils import evaluate_wikitext
from whittle.metrics.parameters import compute_parameters
from whittle.tutorials.gpt_utils import estimate_loss
from whittle.sampling.random_sampler import RandomSampler

logger = logging.getLogger("WHITTLE")


class WHITTLE(BENCHMARK):
    def __init__(
        self,
        model_name: str,
        metrics: List[str] = [
            "perplexity",
            "latency",
            "parameters",
            "loss",
            "lm_evaluation_harness",
        ],
        tasks: List[str] = ["sciq", "hellaswag", "truthfulqa_mc2", "mmlu"],
        seed: int = None,
        task_context: dict = None,
    ) -> None:
        """
        Initialize a WHITTLE benchmark.

        Args:
            model_name (str): The name of the model to be optimized.
            metrics (List[str]): The list of metrics to be used for the benchmark. Defaults to ["perplexity", "latency", "parameters", "loss", "lm_evaluation_harness"].
            tasks (List[str]): The list of tasks to be used for the benchmark. Defaults to ["sciq", "hellaswag", "truthfulqa_mc2", "mmlu"].
            seed (int): The random seed to be used for the benchmark. Defaults to None.
        """
        self.seed: int = seed
        self.model_name: str = model_name
        self.benchmark_name = "WHITTLE"
        self.metrics: List[str] = metrics
        self.tasks: List[str] = tasks
        self.task_context = task_context
        self.device: str = "cpu" if not torch.cuda.is_available() else "cuda"
        self.supernet_model, self.supernet_tokenizer, self.supernet_config = (
            self._loadSuperNet()
        )

    def _loadSuperNet(self) -> Tuple[GPT, Any, Config]:
        """
        Loads a super network (GPT model) from the Hugging Face model hub and
        returns it as a tuple of the model and the corresponding configuration.

        If the model has not been downloaded yet, it is downloaded from the model hub.
        The model is then loaded with the corresponding configuration and reset as a super network.
        The model is also moved to the device specified in the constructor.

        Args:
            model_name (str): The name of the model to be loaded from the model hub.

        Returns:
            Tuple[GPT, Tokenizer, Config]: A tuple containing the loaded model, its tokenizer and the corresponding configuration.
        """
        if not pathlib.Path(f"checkpoints/{self.model_name}/lit_model.pth").exists():
            download_from_hub(
                repo_id=self.model_name, checkpoint_dir=pathlib.Path("checkpoints/")
            )
        config = Config.from_file(f"checkpoints/{self.model_name}/model_config.yaml")
        config.fix_head_size = True  # If not set specifically it raises an error

        model = GPT(config)
        model.load_state_dict(
            torch.load(f"checkpoints/{self.model_name}/lit_model.pth")
        )
        model.eval()
        model.name_or_path = f"checkpoints/{self.model_name}"
        model.config.model_type = "gpt"
        model.config.tie_embeddings = False
        model.reset_super_network()
        model.to(self.device)

        if model is None:
            raise ValueError(
                f"Could not load model {self.model_name} from Hugging Face model hub."
            )
        if config is None:
            raise ValueError(
                f"Could not load configuration for model {self.model_name} from Hugging Face model hub."
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            f"checkpoints/{self.model_name}/"
        )
        if tokenizer is None:
            raise ValueError(
                f"Could not load tokenizer for model {self.model_name} from Hugging Face model hub."
            )
        return model, tokenizer, config

    def _evaluate_lm_evaluation_harness(self, model: GPT):
        """
        Evaluate a sub-network using the LM evaluation harness.

        Args:
            model (GPT): The sub-network to be evaluated.
            task (str): The task to be evaluated on.

        """
        results = evaluator.simple_evaluate(
            model=model,
            tasks=self.tasks,
            batch_size=8,
            device=self.device,
            random_seed=self.seed,
            numpy_random_seed=self.seed,
            torch_random_seed=self.seed,
            fewshot_random_seed=self.seed,
        )
        return results.get("results", results)

    def _estimate_latency(self, model: GPT, batch_size: int = 8, n_samples: int = 10):
        """
        Estimates the latency in milliseconds of a given GPT model's inference.

        Args:
            model (GPT): The GPT model instance for which latency is to be estimated.

        Returns:
            float: The total inference time in milliseconds.
        """
        input_ids = torch.randint(
            0, model.config.padded_vocab_size, (batch_size, model.max_seq_length)
        ).to(self.device)

        # Warm-up to avoid cold-start effects
        for _ in range(5):
            with torch.no_grad():
                model(input_ids)

        start_time = time.time()
        for _ in range(n_samples):
            with torch.no_grad():
                model(input_ids)
        total_time = time.time() - start_time

        # Calculate average latency in milliseconds
        avg_latency_ms = (total_time / n_samples) * 1000
        return avg_latency_ms

    def _get_loss_estimate(self, model: GPT, eval_iterations: int = 100) -> float:
        """
        Estimates the validation loss for a given GPT model.

        Args:
            model (GPT): The GPT model for which the loss is to be estimated.

        Returns:
            float: The estimated validation loss.
        """
        # TODO: For some reason small models have lower loss? Check if this is correct
        loss_estimate = estimate_loss(model, eval_iterations)["valid"].item()
        return loss_estimate

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        model_constraints = self.task_context["hyperparameter_constraints"]

        initial_sub_networks = []
        for _ in range(n_points):
            sub_network_n_layers = random.choice(
                model_constraints["sub_network_n_layers"]
            )
            sub_network_n_embd = random.choice(model_constraints["sub_network_n_embd"])
            sub_network_intermediate_size = [
                random.choice(model_constraints["sub_network_intermediate_size"])
                for _ in range(sub_network_n_layers)
            ]
            sub_network_num_heads = [
                random.choice(model_constraints["sub_network_num_heads"])
                for _ in range(sub_network_n_layers)
            ]

            config = {
                "sub_network_n_embd": sub_network_n_embd,
                "sub_network_intermediate_size": sub_network_intermediate_size,
                "sub_network_num_heads": sub_network_num_heads,
                "sub_network_n_layers": sub_network_n_layers,
            }
            initial_sub_networks.append(config)

        return initial_sub_networks

    def evaluate_point(self, point, **kwargs) -> float:
        evaluations = {}
        logger.debug(f"Evaluating point: {point}")
        for metric in self.metrics:
            self.supernet_model.eval()
            self.supernet_model.set_sub_network(
                sub_network_n_embd=point["sub_network_n_embd"],
                sub_network_intermediate_size=point["sub_network_intermediate_size"],
                sub_network_num_heads=point["sub_network_num_heads"],
                sub_network_n_layers=point["sub_network_n_layers"],
            )  # Set the supernet to the subnetwork
            match metric:
                case "perplexity":
                    evaluations[metric] = evaluate_wikitext(
                        self.supernet_model, self.supernet_tokenizer
                    )
                case "parameters":
                    evaluations[metric] = compute_parameters(self.supernet_model)
                case "loss":
                    evaluations[metric] = self._get_loss_estimate(self.supernet_model)
                case "latency":
                    evaluations[metric] = self._estimate_latency(self.supernet_model)
                case "lm_evaluation_harness":
                    evaluations[metric] = self._evaluate_lm_evaluation_harness(
                        self.supernet_model
                    )
                case _:
                    logger.debug(f"Unknown metric: {metric}")
                    raise ValueError(f"Unknown metric {metric}")

        self.supernet_model.reset_super_network()
        return point, evaluations

    def get_few_shot_samples(self, **kwargs) -> List[Tuple[Dict, Dict]]:
        model_constraints = self.task_context["hyperparameter_constraints"]
        n_embed_middle_index = int(len(model_constraints["sub_network_n_embd"]) / 2)
        n_layers_middle_index = int(len(model_constraints["sub_network_n_layers"]) / 2)
        num_heads_middle_index = int(
            len(model_constraints["sub_network_num_heads"]) / 2
        )
        intermediate_size_middle_index = int(
            len(model_constraints["sub_network_intermediate_size"]) / 2
        )

        largest_model = {
            "sub_network_n_layers": sorted(model_constraints["sub_network_n_layers"])[
                -1
            ],
            "sub_network_n_embd": sorted(model_constraints["sub_network_n_embd"])[-1],
            "sub_network_intermediate_size": [
                sorted(model_constraints["sub_network_intermediate_size"])[-1]
                for _ in range(sorted(model_constraints["sub_network_n_layers"])[-1])
            ],
            "sub_network_num_heads": [
                sorted(model_constraints["sub_network_num_heads"])[-1]
                for _ in range(sorted(model_constraints["sub_network_n_layers"])[-1])
            ],
        }
        middle_model = {
            "sub_network_n_layers": sorted(model_constraints["sub_network_n_layers"])[
                n_layers_middle_index
            ],
            "sub_network_n_embd": sorted(model_constraints["sub_network_n_embd"])[
                n_embed_middle_index
            ],
            "sub_network_intermediate_size": [
                sorted(model_constraints["sub_network_intermediate_size"])[
                    intermediate_size_middle_index
                ]
                for _ in range(
                    sorted(model_constraints["sub_network_n_layers"])[
                        n_layers_middle_index
                    ]
                )
            ],
            "sub_network_num_heads": [
                sorted(model_constraints["sub_network_num_heads"])[
                    num_heads_middle_index
                ]
                for _ in range(
                    sorted(model_constraints["sub_network_n_layers"])[
                        n_layers_middle_index
                    ]
                )
            ],
        }
        smallest_model = {
            "sub_network_n_layers": sorted(model_constraints["sub_network_n_layers"])[
                0
            ],
            "sub_network_n_embd": sorted(model_constraints["sub_network_n_embd"])[0],
            "sub_network_intermediate_size": [
                sorted(model_constraints["sub_network_intermediate_size"])[0]
                for _ in range(sorted(model_constraints["sub_network_n_layers"])[0])
            ],
            "sub_network_num_heads": [
                sorted(model_constraints["sub_network_num_heads"])[0]
                for _ in range(sorted(model_constraints["sub_network_n_layers"])[0])
            ],
        }
        few_shot_examples = [
            (largest_model, self.evaluate_point(largest_model)[1]),
            (middle_model, self.evaluate_point(middle_model)[1]),
            (smallest_model, self.evaluate_point(smallest_model)[1]),
        ]
        return few_shot_examples

    def get_metrics_ranges(self, **kwargs) -> Dict[str, List[float]]:
        few_shot_examples = self.get_few_shot_samples()
        evaluations = pd.DataFrame([evaluation for _, evaluation in few_shot_examples])
        metrics_ranges = {
            metric: [evaluations[metric].min(), evaluations[metric].max()]
            for metric in evaluations.columns
        }
        return metrics_ranges

    def is_valid_candidate(self, candidate):
        model_constraints = self.task_context["hyperparameter_constraints"]
        try:
            # Check if all keys are present
            are_keys_present = (
                "sub_network_n_embd" in candidate.keys()
                and "sub_network_intermediate_size" in candidate.keys()
                and "sub_network_num_heads" in candidate.keys()
                and "sub_network_n_layers" in candidate.keys()
            )
            if not are_keys_present:
                logger.warning("Cannot evaluate candidate: wrong keys")
                return False

            # Check if all values are proper types
            is_valid_type = (
                isinstance(candidate["sub_network_n_embd"], int)
                and isinstance(candidate["sub_network_intermediate_size"], list)
                and isinstance(candidate["sub_network_num_heads"], list)
                and isinstance(candidate["sub_network_n_layers"], int)
            )
            if not is_valid_type:
                logger.warning("Cannot evaluate candidate: wrong type")
                return False

            is_embed_dim_valid = (
                candidate["sub_network_n_embd"]
                in model_constraints["sub_network_n_embd"]
            )

            is_num_layers_valid = (
                candidate["sub_network_n_layers"]
                in model_constraints["sub_network_n_layers"]
            )
            is_layers_size_valid = (
                len(candidate["sub_network_intermediate_size"])
                == len(candidate["sub_network_num_heads"])
                == candidate["sub_network_n_layers"]
            )
            is_num_heads_valid = all(
                n_head in model_constraints["sub_network_num_heads"]
                for n_head in candidate["sub_network_num_heads"]
            )

            if not is_embed_dim_valid:
                logger.warning("Invalid configuration: wrong embed dim")
                return False

            if not is_num_layers_valid:
                logger.warning("Invalid configuration: wrong number of layers")
                return False

            if not is_layers_size_valid:
                logger.warning("Invalid configuration: wrong number of layers")
                return False

            if not is_num_heads_valid:
                logger.warning("Invalid configuration: wrong number of heads")
                return False

            return True
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
        fvals = statistics["observed_fvals"]
        hw_metric = [metric for metric in self.metrics if metric != "perplexity"][0]
        device = "test_device"

        fval_dir = f"./baselines/{self.benchmark_name}/gpt-test/{hw_metric}"
        fval_filename = f"gpt-test_{hw_metric}_{device}_{self.seed}.csv"

        os.makedirs(fval_dir, exist_ok=True)
        fvals.to_csv(f"{fval_dir}/{fval_filename}", index=False)

        logger.debug(f"Writing fvals to {fval_dir}/{fval_filename}")
