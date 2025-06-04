import json
import inspect
import logging
import random

from typing import Dict
from random import randrange, uniform, choice
from mooLLM.llm.llm import LLMInterface

logger = logging.getLogger("Dummy_LLM")


class DUMMY_HWGPT(LLMInterface):
    def __init__(self, model: str = "", forceCPU: bool = False) -> None:
        super().__init__()
        self.model = model

        self.config_choices = {
            "embed_dim_choices": [192, 384, 768],
            "n_layer_choices": [10, 11, 12],
            "mlp_ratio_choices": [2, 3, 4],
            "n_head_choices": [4, 8, 12],
            "bias_choices": ["True", "False"],
        }

    def prompt(self, prompt: str, max_number_of_tokens: str = 100, **kwargs) -> Dict:
        stack = inspect.stack()
        called_by = stack[1][0].f_locals["self"].__class__.__name__

        response = ""
        match called_by:
            case "LLM_SAMPLER":
                response = json.dumps(self.generate_random_configuration())
            case "SURROGATE_MODEL":
                response = f'{{"Latency": {uniform(20, 154.3)},"Perplexity": {uniform(22.12, 78.33)}}}'
            case "ZERO_SHOT_WARMSTARTER":
                response = json.dumps(
                    [
                        self.generate_random_configuration()
                        for _ in range(self.initial_samples)
                    ]
                )
            case _:
                response = ""

        return response

    def generate_random_configuration(self) -> Dict:
        """Generates a random configuration based on the specified choices."""
        # Sample values randomly from each choice list
        embed_dim = choice(self.config_choices["embed_dim_choices"])
        n_layer = choice(self.config_choices["n_layer_choices"])
        layer_indices = [choice(range(n_layer)) for _ in range(n_layer)]
        mlp_ratio = [
            choice(self.config_choices["mlp_ratio_choices"]) for _ in range(n_layer)
        ]

        # Add a small chance that the generation fails
        if random.random() < 0.3:
            n_head = [choice(range(1, 15)) for _ in range(n_layer)]
        else:
            n_head = [
                choice(self.config_choices["n_head_choices"]) for _ in range(n_layer)
            ]

        sample_bias = choice(self.config_choices["bias_choices"])

        return {
            "sample_embed_dim": embed_dim,
            "sample_n_layer": n_layer,
            "sample_layer_indices": layer_indices,
            "sample_n_head": n_head,
            "sample_mlp_ratio": mlp_ratio,
            "sample_bias": sample_bias,
        }
