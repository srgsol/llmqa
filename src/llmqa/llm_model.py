"""LLM Model abstraction for standardized LLM interactions.

This module provides the LLMModel class that serves as an abstraction layer
between the application and various LLM providers.

Example:
    Basic usage with OpenAI:

        from llmqa.providers.openai import OpenAIProvider
        from llmqa.llm_model import LLMModel
        from llmqa.message import Message
        from llmqa.enums import ChatRole

        provider = OpenAIProvider(model="gpt-4o")
        model = LLMModel(provider=provider)

        response = model.generate(
            messages=[Message(role=ChatRole.USER, content="Hello")],
            system_prompt=Message(role=ChatRole.SYSTEM, content="Be helpful")
        )
"""

import logging
import time
from collections.abc import Callable

from .enums import ChatRole
from .error import AuthenticationError, RateLimitError
from .message import Message
from .provider import LLMProvider

log = logging.getLogger(__name__)

# The lower bound for the initial delay in seconds.
MIN_BACKOFF_INITIAL_DELAY_S = 2.0


class LLMModel:
    """Abstraction layer for LLM interactions.

    This class standardizes how the application interacts with different
    LLM providers, handling message formatting and retry logic.

    Attributes:
        provider: The LLM provider to use for API calls.
        enable_rate_limit_backoff: Whether to retry on rate limits.
        backoff_max_retries: Maximum retry attempts.
        backoff_initial_delay_s: Initial delay before first retry.
        backoff_factor: Exponential factor for retry delays.
        backoff_max_delay_s: Maximum delay between retries.

    Example:
        Using with retry configuration:

            model = LLMModel(
                provider=OpenAIProvider(model="gpt-4o"),
                enable_rate_limit_backoff=True,
                backoff_max_retries=5,
            )
    """

    def __init__(
        self,
        provider: LLMProvider,
        enable_rate_limit_backoff: bool = True,
        backoff_max_retries: int = 5,
        backoff_initial_delay_s: float = 8.0,
        backoff_factor: float = 2.0,
        backoff_max_delay_s: float = 256.0,
    ):
        """Initialize the LLM model.

        Args:
            provider: The LLM provider to use.
            enable_rate_limit_backoff: Whether to retry on rate limits.
            backoff_max_retries: Maximum retry attempts.
            backoff_initial_delay_s: Initial delay in seconds before first retry.
            backoff_factor: Exponential factor applied to delay between retries.
            backoff_max_delay_s: Maximum delay in seconds between retries.

        Raises:
            ValueError: If backoff configuration is invalid.
        """
        self.provider = provider
        self.enable_rate_limit_backoff = enable_rate_limit_backoff
        self.backoff_max_retries = backoff_max_retries
        self.backoff_initial_delay_s = backoff_initial_delay_s
        self.backoff_factor = backoff_factor
        self.backoff_max_delay_s = backoff_max_delay_s

        if enable_rate_limit_backoff:
            self._validate_backoff_config()

    def _validate_backoff_config(self) -> None:
        """Validate backoff configuration parameters."""
        errors: list[str] = []
        if self.backoff_max_retries <= 0:
            errors.append(f"backoff_max_retries must be positive (got {self.backoff_max_retries})")
        if self.backoff_initial_delay_s < MIN_BACKOFF_INITIAL_DELAY_S:
            errors.append(
                f"backoff_initial_delay_s must be at least {MIN_BACKOFF_INITIAL_DELAY_S} "
                f"(got {self.backoff_initial_delay_s})"
            )
        if self.backoff_factor < 1:
            errors.append(f"backoff_factor must be at least 1 (got {self.backoff_factor})")
        if self.backoff_max_delay_s <= 0:
            errors.append(f"backoff_max_delay_s must be positive (got {self.backoff_max_delay_s})")

        if errors:
            raise ValueError(f"Invalid backoff configuration: {'; '.join(errors)}")

    def generate(
        self,
        messages: list[Message],
        system_prompt: Message | None = None,
    ) -> list[Message]:
        """Generate responses for the given messages.

        Args:
            messages: List of messages to generate a response for.
            system_prompt: Optional system prompt to prepend to messages.

        Returns:
            List of generated response messages.

        Raises:
            ValueError: If messages list is empty.
        """
        if len(messages) == 0:
            raise ValueError("Messages list cannot be empty")

        # Convert messages to API format
        api_messages = self._format_messages(messages)
        system_prompt_str = system_prompt.content if system_prompt else None

        if self.enable_rate_limit_backoff:
            response = self._call_with_backoff(
                lambda: self.provider.call_api(api_messages, system_prompt=system_prompt_str)
            )
        else:
            response = self.provider.call_api(api_messages, system_prompt=system_prompt_str)

        return [response]

    def _format_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to API format.

        Args:
            messages: List of Message objects.

        Returns:
            List of dictionaries with 'role' and 'content' keys.
        """
        api_messages = []
        for msg in messages:
            if msg.role in [ChatRole.USER, ChatRole.ASSISTANT]:
                api_messages.append({
                    "role": msg.role.value.lower(),
                    "content": msg.content,
                })
            elif msg.role == ChatRole.SYSTEM:
                api_messages.append({
                    "role": "system",
                    "content": msg.content,
                })
            elif msg.role == ChatRole.FUNCTION:
                # Function messages are typically handled as assistant messages
                api_messages.append({
                    "role": "assistant",
                    "content": msg.content,
                })
        return api_messages

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry with backoff.

        Args:
            error: The exception raised by the provider call.

        Returns:
            True if the error is considered transient and should be retried.
        """
        if isinstance(error, AuthenticationError | ValueError | RuntimeError):
            return False
        return isinstance(error, RateLimitError)

    def _call_with_backoff(self, func: Callable[[], Message]) -> Message:
        """Execute a provider call with optional exponential backoff.

        Args:
            func: Zero-argument function that performs the provider call.

        Returns:
            Provider response message.

        Raises:
            Exception: Re-raises the last exception if not retryable or retries exhausted.
        """
        sleep_delay = self.backoff_initial_delay_s
        for _ in range(self.backoff_max_retries):
            try:
                return func()
            except Exception as exc:
                if not self._is_retryable_error(exc):
                    raise
                log.warning(f"Rate limit hit, retrying in {sleep_delay}s: {exc}")
                time.sleep(sleep_delay)
                sleep_delay = min(self.backoff_max_delay_s, sleep_delay * self.backoff_factor)

        # Final attempt without catching exceptions.
        return func()
