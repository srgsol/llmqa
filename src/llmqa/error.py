"""Custom exceptions for the llmqa testing framework.

This module defines exceptions used throughout the framework for
error handling and reporting.
"""


class LLMQAError(Exception):
    """Base exception for all llmqa errors."""

    pass


class ProviderError(LLMQAError):
    """Error raised when an LLM provider fails."""

    pass


class AuthenticationError(ProviderError):
    """Error raised when authentication fails."""

    pass


class RateLimitError(ProviderError):
    """Error raised when rate limits are exceeded."""

    pass


class AgentError(LLMQAError):
    """Error raised when an agent fails."""

    pass


class AgentGenerationError(AgentError):
    """Error raised when agent generation fails."""

    def __init__(self, message: str, generated_messages: list | None = None):
        """Initialize the error.

        Args:
            message: Error description.
            generated_messages: Messages generated before the error occurred.
        """
        super().__init__(message)
        self.generated_messages = generated_messages or []


class FunctionCallParsingError(LLMQAError):
    """Error raised when function call parsing fails."""

    pass
