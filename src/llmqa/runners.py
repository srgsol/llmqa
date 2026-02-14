"""Runners for testing LLM agents.

This module provides test runners for executing interactions with LLM agents
and validating their responses using expectations.

Example:
    Basic usage with ConversationalAgentRunner:

        from llmqa.runners import ConversationalAgentRunner
        from llmqa.expectations import TextContains
        from llmqa.agent import AgentQA

        agent = AgentQA(model="gpt-4o")
        runner = ConversationalAgentRunner(agent)

        interactions = [
            {
                "input": "What is Python?",
                "expectations": [TextContains(text="programming")],
            }
        ]

        runner.assert_interactions(interactions)
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from .agent import Agent
from .enums import ChatRole
from .expectations import BaseExpectation
from .message import Message
from .utils import render_conversation

logger = logging.getLogger(__name__)


class Interaction(BaseModel):
    """An interaction with the agent.

    An interaction is a single exchange between the user and the agent,
    consisting of an input message and a list of expectations to validate
    the response.

    Attributes:
        input: The input message or list of messages to send to the agent.
        expectations: A list of expectations to run on the response.

    Example:
        >>> from llmqa.expectations import TextContains
        >>> interaction = Interaction(
        ...     input="Hello, how are you?",
        ...     expectations=[TextContains(text="Hello")],
        ... )
    """

    input: str | Message | list[Message]
    expectations: Sequence[BaseExpectation]


def assert_expectations(
    expectations: Sequence[BaseExpectation],
    response: list[Message],
) -> None:
    """Assert a list of expectations against a response.

    Args:
        expectations: List of expectations to validate.
        response: List of messages to validate against.

    Raises:
        AssertionError: If any expectation fails.
    """
    for expectation in expectations:
        expectation.run(response)


def build_input_messages(
    interaction_input: str | Message | list[Message],
) -> list[Message]:
    """Build a list of input messages from an interaction input.

    Args:
        interaction_input: The input of the interaction. Can be a string,
            a single Message, or a list of Messages.

    Returns:
        A list of Message objects.
    """
    if isinstance(interaction_input, str):
        return [Message(role=ChatRole.USER, content=interaction_input)]
    if isinstance(interaction_input, Message):
        return [interaction_input]
    return interaction_input


def _validate_interaction(interaction: dict[str, Any] | Interaction) -> Interaction:
    """Validate an interaction.

    Args:
        interaction: An interaction to validate (dict or Interaction).

    Returns:
        An Interaction object.

    Raises:
        pydantic.ValidationError: If the interaction is not in the correct format.
    """
    if isinstance(interaction, Interaction):
        return interaction
    # Validate interaction formatted as dict.
    # If format is wrong pydantic will raise a validation error.
    return Interaction(**interaction)


def _validate_interactions(
    interactions: list[dict[str, Any] | Interaction],
) -> list[Interaction]:
    """Validate a list of interactions.

    Args:
        interactions: A list of potentially mixed format interactions to validate.

    Returns:
        A list of validated Interaction objects.
    """
    return [_validate_interaction(interaction) for interaction in interactions]


def assert_interactions(
    agent: Agent,
    interactions: list[dict[str, Any] | Interaction],
    log_response: bool = False,
) -> None:
    """Assert a list of interactions with an agent.

    Each interaction is independent - no conversation history is maintained
    between interactions.

    Args:
        agent: The agent to run the interactions with.
        interactions: A list of interactions to run.
        log_response: Whether to log the response.

    Example:
        >>> from llmqa.agent import AgentQA
        >>> from llmqa.expectations import TextContains
        >>> agent = AgentQA()
        >>> interactions = [
        ...     {"input": "Hello", "expectations": [TextContains(text="Hi")]},
        ... ]
        >>> assert_interactions(agent, interactions)
    """
    validated_interactions = _validate_interactions(interactions)
    for interaction in validated_interactions:
        input_msg = build_input_messages(interaction.input)
        response = agent.generate(input_msg)
        if log_response:
            logger.info(f"Response: {response}")
        assert_expectations(interaction.expectations, response)


class AbstractConversationalRunner(ABC):
    """Abstract base class for conversational runners.

    This class provides the base interface for all conversational runners
    that maintain conversation history across interactions.
    """

    def __init__(self) -> None:
        """Initialize the conversational runner."""
        self.conversation_history: list[Message] = []
        self.log_conversation: bool = False
        self.log_conversation_on_error: bool = True

    def assert_interactions(
        self,
        interactions: list[dict[str, Any] | Interaction],
        log_conversation: bool = False,
        log_conversation_on_error: bool = True,
    ) -> None:
        """Assert a list of interactions.

        Args:
            interactions: A list of interactions to assert.
            log_conversation: Whether to log the conversation history after
                all interactions complete successfully.
            log_conversation_on_error: Whether to log the conversation history
                when an error occurs.
        """
        validated_interactions = _validate_interactions(interactions)
        self.log_conversation = log_conversation
        self.log_conversation_on_error = log_conversation_on_error
        try:
            self._assert_interactions(validated_interactions)
        except Exception as e:
            if self.log_conversation_on_error:
                logger.error(self.build_log_conversation())
            raise e
        if self.log_conversation:
            logger.info(self.build_log_conversation())

    def build_log_conversation(self) -> str:
        """Build a string representation of the conversation history.

        Returns:
            A formatted string representation of the conversation history.
        """
        return render_conversation(self.conversation_history)

    def build_input_messages(self, interaction: Interaction) -> list[Message]:
        """Build a list of input messages from an interaction.

        Args:
            interaction: The interaction to build the input messages from.

        Returns:
            A list of Message objects.
        """
        return build_input_messages(interaction.input)

    @abstractmethod
    def _assert_interactions(self, interactions: list[Interaction]) -> None:
        """Run a list of interactions.

        Args:
            interactions: A list of interactions to run.
        """
        pass


class ConversationalAgentRunner(AbstractConversationalRunner):
    """Runner that maintains a conversation with an Agent.

    This runner exchanges messages with the Agent, maintaining a full
    conversation history across interactions.

    At each interaction, the runner sends the full conversation history
    to the Agent. Then it appends the Agent's response to the history
    and runs expectations on that response.

    The `conversation_history` is kept for the entire agent lifetime.
    It accumulates all the messages exchanged with the Agent in different runs.

    Example:
        >>> from llmqa.agent import AgentQA
        >>> from llmqa.expectations import TextContains
        >>> agent = AgentQA(model="gpt-4o")
        >>> runner = ConversationalAgentRunner(agent)
        >>> interactions = [
        ...     {
        ...         "input": "What is the capital of France?",
        ...         "expectations": [TextContains(text="Paris")],
        ...     },
        ...     {
        ...         "input": "What is its best known monument?",
        ...         "expectations": [TextContains(text="Eiffel")],
        ...     },
        ... ]
        >>> runner.assert_interactions(interactions)
        >>> print(len(runner.conversation_history))  # 4 messages
    """

    def __init__(self, agent: Agent) -> None:
        """Initialize the conversational agent runner.

        Args:
            agent: The agent to run the interactions with.
        """
        self.agent = agent
        super().__init__()

    def _assert_interactions(self, interactions: list[Interaction]) -> None:
        """Run a list of interactions with conversation history.

        This method implements a conversation loop that maintains a conversation
        history across all interactions and checks the expectations on the
        response at each interaction.

        For each interaction, the method:
        - Builds the input messages.
        - Generates the Agent's response.
        - Adds the Agent's response to the conversation history.
        - Runs expectations on the response.
        - Raises an exception if any expectation is not met.

        Args:
            interactions: A list of Interaction objects to run.
        """
        for interaction in interactions:
            input_msg = self.build_input_messages(interaction)
            self.conversation_history.extend(input_msg)

            # Generate Agent's response
            response = self.agent.generate(self.conversation_history)

            # Add agent response to conversation history
            self.conversation_history.extend(response)

            # Run expectations on the response
            assert_expectations(interaction.expectations, response)
