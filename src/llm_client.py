"""
OpenRouter LLM client for semantic analysis.
"""
import json
import logging
import time
from typing import Optional

from openai import OpenAI

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client using OpenRouter API.
    Supports multiple models through a unified interface.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier (e.g., "anthropic/claude-sonnet-4")
        """
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Create a .env file with your API key or set the environment variable."
            )

        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/kirjakeel-konekeel",
                "X-Title": "Kirjakeel-Konekeel Analysis Tool",
            },
        )
        self.model = model
        self._request_count = 0
        self._last_request_time = 0

    def set_model(self, model: str):
        """
        Switch to a different model.

        Args:
            model: Model identifier
        """
        if model not in AVAILABLE_MODELS:
            print(f"Warning: {model} not in known models, but attempting anyway")
        self.model = model

    def _rate_limit(self, min_interval: float = 0.5):
        """Simple rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def analyze(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """
        Send a prompt to the LLM and get a response.

        Args:
            prompt: The instruction/question
            context: Optional context to include
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum response length

        Returns:
            LLM response text
        """
        self._rate_limit()

        messages = []

        if context:
            messages.append({
                "role": "user",
                "content": f"{prompt}\n\nContext:\n{context}"
            })
        else:
            messages.append({
                "role": "user",
                "content": prompt
            })

        try:
            logger.debug(f"LLM request #{self._request_count + 1} to {self.model}")
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._request_count += 1

            elapsed = time.time() - start_time
            result = response.choices[0].message.content.strip()
            logger.debug(f"LLM response received in {elapsed:.2f}s ({len(result)} chars)")

            return result

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def analyze_batch(
        self,
        prompts: list[tuple[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        show_progress: bool = True,
    ) -> list[str]:
        """
        Process multiple prompts sequentially.

        Args:
            prompts: List of (prompt, context) tuples
            temperature: Sampling temperature
            max_tokens: Maximum response length per request
            show_progress: Whether to print progress

        Returns:
            List of responses
        """
        results = []
        total = len(prompts)

        for i, (prompt, context) in enumerate(prompts):
            if show_progress:
                print(f"Processing {i+1}/{total}...", end="\r")

            try:
                result = self.analyze(prompt, context, temperature, max_tokens)
                results.append(result)
            except Exception as e:
                results.append(f"ERROR: {e}")

        if show_progress:
            print(f"Completed {total}/{total} requests")

        return results

    def analyze_json(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.1,
    ) -> Optional[dict]:
        """
        Get a JSON response from the LLM.

        Args:
            prompt: The instruction (should ask for JSON output)
            context: Optional context
            temperature: Lower for more consistent JSON

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        response = self.analyze(prompt, context, temperature)

        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end]
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end]

            return json.loads(response.strip())

        except json.JSONDecodeError:
            # Try to find JSON object in response
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        return None

    @property
    def request_count(self) -> int:
        """Number of requests made in this session."""
        return self._request_count

    def get_available_models(self) -> list[str]:
        """Return list of known available models."""
        return AVAILABLE_MODELS.copy()


class MockLLMClient:
    """
    Mock LLM client for testing without API calls.
    """

    def __init__(self, model: str = "mock/model"):
        self.model = model
        self._request_count = 0

    def set_model(self, model: str):
        self.model = model

    def analyze(self, prompt: str, context: str = "", **kwargs) -> str:
        self._request_count += 1
        return f"[Mock response for: {prompt[:50]}...]"

    def analyze_batch(self, prompts: list[tuple[str, str]], **kwargs) -> list[str]:
        return [self.analyze(p, c) for p, c in prompts]

    def analyze_json(self, prompt: str, context: str = "", **kwargs) -> dict:
        self._request_count += 1
        return {"mock": True, "response": "test"}

    @property
    def request_count(self) -> int:
        return self._request_count

    def get_available_models(self) -> list[str]:
        return ["mock/model"]


def get_llm_client(model: str = DEFAULT_MODEL, mock: bool = False) -> LLMClient | MockLLMClient:
    """
    Get an LLM client instance.

    Args:
        model: Model to use
        mock: If True, return a mock client for testing

    Returns:
        LLM client instance
    """
    if mock:
        return MockLLMClient(model)
    return LLMClient(model)
