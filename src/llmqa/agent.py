"""Agent implementations for LLM-based QA testing.

This module provides the base Agent class and the AgentQA implementation
for quality assurance testing of LLM responses.

Example:
    Basic usage with AgentQA:

        from llmqa.agent import AgentQA
        from llmqa.message import Message
        from llmqa.enums import ChatRole

        agent = AgentQA(model="gpt-4o")
        messages = [Message(role=ChatRole.USER, content="What is Python?")]
        response = agent.generate(messages)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from .enums import ChatRole
from .error import AgentGenerationError
from .llm_model import LLMModel
from .message import Message
from .providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are Quality Assurance Agent. Your task is to check the quality of the answers \
provided by an LLM under test.

{test_system_prompt}
"""


class Agent(ABC):
    """Abstract base class defining the agent interface.

    This class provides the base interface for all agent implementations
    that handle LLM interactions.

    Attributes:
        llm_model: LLM implementation for generating responses.
        system_prompt: Initial prompt that guides the LLM's behavior.
    """

    def __init__(
        self,
        llm_model: LLMModel,
        system_prompt: str,
    ):
        """Initialize the agent.

        Args:
            llm_model: LLM implementation to use for generating responses.
            system_prompt: Initial prompt that guides the LLM's behavior.
        """
        self.llm_model = llm_model
        self.system_prompt = system_prompt

    @abstractmethod
    def get_system_prompt(self, context_args: dict[str, Any] | None = None) -> Message:
        """Get the system prompt as a Message.

        Args:
            context_args: Optional context arguments for prompt formatting.

        Returns:
            Message containing the system prompt.
        """
        ...  # pragma: no cover

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        context_args: dict[str, Any] | None = None,
    ) -> list[Message]:
        """Generate responses through LLM interactions.

        Args:
            messages: The list of messages to generate responses for.
            context_args: Additional context arguments.

        Returns:
            The list of messages generated during this interaction.
        """
        ...  # pragma: no cover


class AgentQA(Agent):
    """Quality Assurance Agent using OpenAI.

    This agent is designed for testing LLM responses. It uses OpenAI as the
    underlying LLM provider.

    Attributes:
        llm_model: Inherited from Agent, handles LLM interactions.
        system_prompt: Inherited from Agent, guides LLM behavior.

    Example:
        Basic usage:

            from llmqa.agent import AgentQA
            from llmqa.message import Message
            from llmqa.enums import ChatRole

            agent = AgentQA(model="gpt-4o")
            messages = [Message(role=ChatRole.USER, content="Hello")]
            response = agent.generate(messages)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
    ):
        """Initialize the QA agent.

        Args:
            model: The OpenAI model to use (default: "gpt-4o").
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate per response.
            system_prompt: Custom system prompt. If None, uses default QA prompt.
        """
        logger.info(f"Instantiating {self.__class__.__name__} ({model})")

        try:
            provider = OpenAIProvider(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            llm_model = LLMModel(provider=provider)

            super().__init__(
                llm_model=llm_model,
                system_prompt=system_prompt or SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise RuntimeError(f"Failed to initialize LLM model: {e}") from e

    def get_system_prompt(self, context_args: dict[str, Any] | None = None) -> Message:
        """Get the system prompt as a Message.

        Args:
            context_args: Optional context arguments (unused in base implementation).

        Returns:
            Message containing the system prompt.
        """
        _ = context_args  # Available for subclasses to use
        return Message(role=ChatRole.SYSTEM, content=self.system_prompt)

    def generate(
        self,
        messages: list[Message],
        context_args: dict[str, Any] | None = None,
    ) -> list[Message]:
        """Generate a response from the agent.

        Args:
            messages: List of messages to respond to.
            context_args: Optional context arguments.

        Returns:
            List of response messages.

        Raises:
            AgentGenerationError: If generation fails.
        """
        try:
            prompt = self.get_system_prompt(context_args)
            llm_response = self.llm_model.generate(messages, system_prompt=prompt)
            logger.debug(f"LLM Response: {llm_response}")
            return llm_response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise AgentGenerationError(f"Failed to generate response: {e}") from e
