import re
import json
import logging
from typing import Dict
from abc import ABC, abstractmethod
from HOLLM.statistics.statistics import Statistics
from HOLLM.utils.estimators import estimate_cost

logger = logging.getLogger("LLMInterface")


class LLMInterface(ABC):
    def __init__(self, model: str = "", forceCPU: bool = False):
        """
        Initializes the LLM Interface.

        Args:
            model (str): The name of the model to use.
            forceCPU (bool): If True, forces the model to run on CPU even if a GPU is available.
        """

        self.initial_samples = None
        self.llm_settings: dict = {}
        self.statistics: Statistics = None

    @abstractmethod
    def prompt(self, prompt: str, max_number_of_tokens: str = 100, **kwargs) -> str:
        """
        Abstract method to send a prompt to the LLM and get the response back.

        Args:
            prompt (str): The prompt to be sent to the LLM.
            max_number_of_tokens (str): The maximum number of tokens to be generated by the LLM as a response.

        Returns:
            str: The response from the LLM.
        """
        pass

    def update_cost_and_token_usage(self, response):
        """Update the cost and token usage statistics based on the LLM response.

        Args:
            response: The response object from the LLM, which should contain token usage information.
        """
        token_usage = response.usage
        cost = estimate_cost(
            {
                "input_cost_per_1000_tokens": self.llm_settings.get(
                    "input_cost_per_1000_tokens", 0
                ),
                "output_cost_per_1000_tokens": self.llm_settings.get(
                    "output_cost_per_1000_tokens", 0
                ),
            },
            token_usage,
        )
        self.statistics.update_cost(cost)
        self.statistics.update_token_usage(token_usage)

    def to_json(self, response: str) -> Dict:
        """
        Converts a model's response string to a JSON dictionary.

        Given a model's response as a string, this method removes any Markdown symbols
        and returns the cleaned JSON string as a dictionary.

        Args:
            response (str): The model's response containing the JSON output.

        Returns:
            Dict: The cleaned JSON string converted to a dictionary.

        """
        logger.debug(f"LLM raw response: {response}")
        cleaned_response = self._clean_json_markdown(response)
        return json.loads(cleaned_response)

    def _clean_json_markdown(self, model_response: str) -> str:
        """
        Removes Markdown symbols from a model's JSON response and returns the cleaned string.

        Args:
            model_response (str): The original model response containing Markdown symbols.

        Returns:
            str: The cleaned JSON string without Markdown symbols.
        """
        # Regex pattern to match the JSON content
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, model_response, re.DOTALL)
        if match:
            logger.debug(f"MODEL RESPONSE MATCH: {match}")
            # Extract the JSON part
            json_part = match.group(1).strip()
            # Replace single quotes with double quotes to conform to JSON format
            # If the example for the LLM is correctly formatted, this should not be needed
            json_part = json_part.replace("'", '"')
            return json_part
        else:
            logger.debug("No JSON match found in the model response.")
            # If no Markdown, check if the response is valid JSON
            return model_response.strip()
