"""Provider interface for LLM interactions.

This module defines the base interface for LLM providers, establishing
a standard way to interact with different LLM services.

Example:
    Custom providers can be implemented by inheriting from LLMProvider:

        from llmqa.provider import LLMProvider
        from llmqa.message import Message

        class CustomProvider(LLMProvider):
            def call_api(self, messages, system_prompt=None) -> Message:
                response = self._make_api_call(messages)
                return Message(role=ChatRole.ASSISTANT, content=response)
"""

import logging
from abc import ABC, abstractmethod

from .message import Message

log = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM service providers.

    This class defines the interface that all LLM providers must implement
    to ensure consistent interaction with different LLM services.

    Example:
        Implementation example using OpenAI:

            class OpenAIProvider(LLMProvider):
                def call_api(self, messages, system_prompt=None) -> Message:
                    response = self.client.chat.completions.create(...)
                    return Message(role=ChatRole.ASSISTANT, content=response)
    """

    @abstractmethod
    def call_api(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> Message:
        """Call the LLM service API and return a formatted response.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            system_prompt: Optional system prompt to guide the model.

        Returns:
            A Message object containing the LLM's response.

        Raises:
            NotImplementedError: When called directly on the base class.
        """
        ...  # pragma: no cover
