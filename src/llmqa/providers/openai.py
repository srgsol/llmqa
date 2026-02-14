"""OpenAI LLM provider implementation.

This module provides the OpenAI provider for making API calls to OpenAI's
chat completion API.

Example:
    Basic usage:

        from llmqa.providers.openai import OpenAIProvider

        provider = OpenAIProvider(model="gpt-4o")
        response = provider.call_api(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="Be helpful"
        )

Dependencies:
    Requires valid OpenAI API key set via OPENAI_API_KEY environment variable
    or passed directly to the constructor.
"""

import logging
import os

from openai import APIError, OpenAI
from openai import AuthenticationError as OpenAIAuthError
from openai import RateLimitError as OpenAIRateLimitError

from ..enums import ChatRole
from ..error import AuthenticationError, ProviderError, RateLimitError
from ..message import Message
from ..provider import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for LLM interactions.

    This provider handles API calls to OpenAI and processes responses into
    standardized Message objects.

    Attributes:
        model: The OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini").
        temperature: Sampling temperature (0.0=deterministic, 2.0=random).
        max_tokens: Maximum number of tokens to generate.
        client: OpenAI client instance.

    Example:
        Using custom parameters:

            provider = OpenAIProvider(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=1000,
            )
            response = provider.call_api(
                messages=[{"role": "user", "content": "Explain quantum computing"}]
            )
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        base_url: str | None = None,
    ):
        """Initialize the OpenAI provider.

        Args:
            model: The OpenAI model to use.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate per response.
            base_url: Optional custom base URL for API calls.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
        )

        logger.info(f"Initialized OpenAIProvider with model={model}")

    def call_api(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> Message:
        """Call OpenAI API and return the response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            system_prompt: Optional system prompt to guide the model.

        Returns:
            Message object containing the model's response.

        Raises:
            AuthenticationError: For invalid API key or permission issues.
            RateLimitError: For rate limit or quota exceeded errors.
            ProviderError: For other API errors.
        """
        # Build the messages list for the API call
        api_messages: list[dict[str, str]] = []

        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        api_messages.extend(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,  # type: ignore
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content or ""

            return Message(
                role=ChatRole.ASSISTANT,
                content=content,
            )

        except OpenAIAuthError as e:
            logger.error(f"OpenAI authentication error: {e}")
            raise AuthenticationError(f"Authentication failed: {e}") from e

        except OpenAIRateLimitError as e:
            logger.warning(f"OpenAI rate limit error: {e}")
            raise RateLimitError(f"Rate limit exceeded: {e}") from e

        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ProviderError(f"API error: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise ProviderError(f"Unexpected error: {e}") from e
