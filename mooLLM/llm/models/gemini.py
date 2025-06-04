import logging
import google.generativeai as genai
from mooLLM.llm.llm import LLMInterface
import os

logger = logging.getLogger("GEMINI")


class GEMINI(LLMInterface):
    def __init__(self, model: str = "", forceCPU: bool = False):
        super().__init__()

        api_key = os.environ.get("GOOGLE_AI_API_KEY")
        if not api_key:
            logger.debug(
                "No API key found. Please set the GOOGLE_AI_API_KEY environment variable"
            )

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.model = model
        self.rate_limiter = None

    def prompt(self, prompt: str, max_number_of_tokens: str = 100, **kwargs) -> str:
        self.rate_limiter.add_request(request_text=prompt)
        response = self.client.generate_content(prompt)
        response_text = response.text
        request_token_count = response.usage_metadata.total_token_count
        self.rate_limiter.add_request(request_token_count=request_token_count)
        # TODO: Add the cost and token usage here
        return response_text
