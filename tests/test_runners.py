"""Tests for runners."""

import logging
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from llmqa.agent import Agent
from llmqa.enums import ChatRole
from llmqa.expectations import BaseExpectation
from llmqa.llm_model import LLMModel
from llmqa.message import Message
from llmqa.runners import (
    AbstractConversationalRunner,
    ConversationalAgentRunner,
    Interaction,
    _validate_interactions,
    assert_expectations,
    assert_interactions,
    build_input_messages,
)


class MockPassingExpectation(BaseExpectation):
    """Mock passing expectation."""

    def _run(self, _response: list[Message]) -> None:
        """Run the expectation."""
        assert True


class MockFailingExpectation(BaseExpectation):
    """Mock failing expectation."""

    def _run(self, _response: list[Message]) -> None:
        """Run the expectation."""
        raise AssertionError("Test expectation failed")


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, responses: list[list[Message]] | None = None):
        """Initialize mock agent with predefined responses.

        Args:
            responses: List of response lists. Each call to generate() returns the next response.
        """
        mock_llm = Mock(spec=LLMModel)
        super().__init__(llm_model=mock_llm, system_prompt="Test system prompt")
        self.responses = responses or []
        self.call_count = 0
        self.received_messages: list[list[Message]] = []

    def get_system_prompt(self, context_args: dict[str, Any] | None = None) -> Message:
        """Return a test system prompt."""
        _ = context_args  # linter requirement
        return Message(role=ChatRole.SYSTEM, content=self.system_prompt)

    def generate(
        self, messages: list[Message], context_args: dict[str, Any] | None = None
    ) -> list[Message]:
        """Generate a response from the agent."""
        _ = context_args  # linter requirement
        self.received_messages.append(messages.copy())
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return [Message(role=ChatRole.ASSISTANT, content="Default response")]


class TestInteraction:
    """Tests for Interaction model."""

    def test_interaction_with_string_input(self):
        """Test Interaction creation with string input."""
        expectations = [MockPassingExpectation()]
        interaction = Interaction(input="Hello, how are you?", expectations=expectations)
        assert interaction.input == "Hello, how are you?"
        assert len(interaction.expectations) == 1

    def test_interaction_with_message_input(self):
        """Test Interaction creation with Message input."""
        message = Message(role=ChatRole.USER, content="Test message")
        expectations = [MockPassingExpectation()]
        interaction = Interaction(input=message, expectations=expectations)
        assert interaction.input == message
        assert len(interaction.expectations) == 1

    def test_interaction_with_list_of_messages(self):
        """Test Interaction creation with list of messages."""
        messages = [
            Message(role=ChatRole.USER, content="Message 1"),
            Message(role=ChatRole.USER, content="Message 2"),
        ]
        expectations = [MockPassingExpectation()]
        interaction = Interaction(input=messages, expectations=expectations)
        assert interaction.input == messages
        assert len(interaction.expectations) == 1


class TestBuildInputMessages:
    """Tests for build_input_messages function."""

    def test_build_input_messages_from_string(self):
        """Test building messages from a string."""
        result = build_input_messages("Hello")
        assert len(result) == 1
        assert isinstance(result, list)
        assert isinstance(result[0], Message)
        assert result[0].role == ChatRole.USER
        assert result[0].content == "Hello"

    def test_build_input_messages_from_message(self):
        """Test building messages from a Message object."""
        message = Message(role=ChatRole.USER, content="Test")
        result = build_input_messages(message)
        assert len(result) == 1
        assert isinstance(result, list)
        assert isinstance(result[0], Message)
        assert result[0] == message

    def test_build_input_messages_from_list(self):
        """Test building messages from a list of Message objects."""
        messages = [
            Message(role=ChatRole.USER, content="Message 1"),
            Message(role=ChatRole.USER, content="Message 2"),
        ]
        result = build_input_messages(messages)
        assert isinstance(result, list)
        assert isinstance(result[0], Message)
        assert isinstance(result[1], Message)
        assert result == messages


class TestAssertExpectations:
    """Tests for assert_expectations function."""

    def test_assert_expectations_run_method_is_called(self):
        """Test assert_expectations when run method is called.

        This is not to check that the expectations pass, but to check that the
        run method is called.
        """
        expectations = [MockPassingExpectation()]
        response = [Message(role=ChatRole.ASSISTANT, content="Hello world")]

        # Mock the _run method to verify it's called
        with patch.object(expectations[0], "_run") as mock_run:
            # Should not raise
            assert_expectations(expectations, response)
            # Verify _run was called with the response
            # Compare only relevant fields since Message objects have dynamic fields
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert len(call_args) == 1
            assert call_args[0].role == ChatRole.ASSISTANT
            assert call_args[0].content == "Hello world"

    def test_assert_expectations_passes_single_expectation(self):
        """Test assert_expectations when all expectations pass."""
        expectations = [MockPassingExpectation()]
        response = [Message(role=ChatRole.ASSISTANT, content="Hello world")]
        # Should not raise
        assert_expectations(expectations, response)

    def test_assert_expectations_passes_multiple_expectations(self):
        """Test assert_expectations when all expectations pass."""
        expectations = [MockPassingExpectation(), MockPassingExpectation()]
        response = [Message(role=ChatRole.ASSISTANT, content="Hello world")]
        # Should not raise
        assert_expectations(expectations, response)

    def test_assert_expectations_fails_single_expectation(self):
        """Test assert_expectations when an expectation fails."""
        # Test with multiple expectations to ensure the list is processed
        # The first expectation passes, the second fails
        expectations = [MockFailingExpectation()]
        response = [Message(role=ChatRole.ASSISTANT, content="Hello world")]
        with pytest.raises(AssertionError, match="Test expectation failed"):
            assert_expectations(expectations, response)

    def test_assert_expectations_fails_multiple_expectations(self):
        """Test assert_expectations when an expectation fails."""
        # Test with multiple expectations to ensure the list is processed
        # The first expectation passes, the second fails
        expectations = [MockFailingExpectation(), MockFailingExpectation()]
        response = [Message(role=ChatRole.ASSISTANT, content="Hello world")]
        with pytest.raises(AssertionError, match="Test expectation failed"):
            assert_expectations(expectations, response)

    def test_assert_expectations_empty_list(self):
        """Test assert_expectations with empty expectations list."""
        expectations: list[BaseExpectation] = []
        response = [Message(role=ChatRole.ASSISTANT, content="Hello world")]
        # Should not raise
        assert_expectations(expectations, response)


class TestValidateInteractions:
    """Tests for _validate_interactions function."""

    def test_validate_interactions_from_dict(self):
        """Test validation of interactions from dict format."""
        # Test with dict containing BaseExpectation objects
        interactions_dict = [
            {
                "input": "Hello",
                "expectations": [MockPassingExpectation()],
            }
        ]
        result = _validate_interactions(interactions_dict)
        assert len(result) == 1
        assert isinstance(result[0], Interaction)
        assert result[0].input == "Hello"
        assert len(result[0].expectations) == 1
        assert isinstance(result[0].expectations[0], MockPassingExpectation)

    def test_validate_interactions_from_interaction_objects(self):
        """Test validation of interactions that are already Interaction objects."""
        interactions = [
            Interaction(
                input="Hello",
                expectations=[MockPassingExpectation()],
            )
        ]
        result = _validate_interactions(interactions)
        assert result == interactions
        assert isinstance(result[0], Interaction)

    def test_validate_interactions_malformed_dict_missing_field(self):
        """Test validation fails with malformed interaction dict missing required field."""
        # Test with missing required 'expectations' field
        interactions_dict = [
            {
                "input": "Hello",
                # Missing 'expectations' field
            }
        ]
        with pytest.raises(ValidationError):
            _validate_interactions(interactions_dict)

    def test_validate_interactions_invalid_expectations_type(self):
        """Test validation fails with invalid expectations type."""
        # Test with invalid expectations type (should be a list)
        interactions_dict = [
            {
                "input": "Hello",
                "expectations": "not a list",
            }
        ]
        with pytest.raises(ValidationError):
            _validate_interactions(interactions_dict)

    def test_validate_interactions_invalid_expectation_items(self):
        """Test validation fails with invalid expectation items."""
        # Test with non-BaseExpectation items in expectations list
        interactions_dict = [
            {
                "input": "Hello",
                "expectations": ["not an expectation object"],
            }
        ]
        with pytest.raises(ValidationError):
            _validate_interactions(interactions_dict)

    def test_validate_interactions_mixed_dict_and_objects(self):
        """Test validation of interactions that are a mix of dicts and Interaction objects."""
        # Test with mix of dict and Interaction objects
        interactions = [
            {
                "input": "Hello",
                "expectations": [MockPassingExpectation()],
            },
            Interaction(
                input="World",
                expectations=[MockPassingExpectation()],
            ),
            {
                "input": "Test",
                "expectations": [MockPassingExpectation()],
            },
        ]
        # Cast to Any to satisfy type checker - the function should handle mixed types
        result = _validate_interactions(cast("Any", interactions))
        assert len(result) == 3  # noqa: PLR2004
        # All items should be Interaction objects
        assert all(isinstance(item, Interaction) for item in result)
        # Verify the converted dicts
        assert result[0].input == "Hello"
        assert len(result[0].expectations) == 1  # noqa: PLR2004
        # Verify the Interaction object remains unchanged
        assert result[1].input == "World"
        assert len(result[1].expectations) == 1  # noqa: PLR2004
        # Verify the second converted dict
        assert result[2].input == "Test"
        assert len(result[2].expectations) == 1  # noqa: PLR2004


class TestAssertInteractions:
    """Tests for assert_interactions function."""

    def test_assert_interactions_single_interaction(self):
        """Test assert_interactions with a single interaction."""
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Hello world")]])
        interactions = [
            Interaction(
                input="Hello",
                expectations=[MockPassingExpectation()],
            )
        ]
        assert_interactions(agent, interactions)
        assert agent.call_count == 1

        interactions = [
            Interaction(
                input="Hello",
                expectations=[MockFailingExpectation()],
            )
        ]
        with pytest.raises(AssertionError):
            assert_interactions(agent, interactions)
        assert agent.call_count == 1
        assert agent.received_messages[0][0].content == "Hello"

    def test_assert_interactions_multiple_interactions(self):
        """Test assert_interactions with multiple interactions."""
        agent = MockAgent(
            responses=[
                [Message(role=ChatRole.ASSISTANT, content="Response 1")],
                [Message(role=ChatRole.ASSISTANT, content="Response 2")],
            ]
        )
        interactions = [
            Interaction(
                input="Input 1",
                expectations=[MockPassingExpectation()],
            ),
            Interaction(
                input="Input 2",
                expectations=[MockPassingExpectation()],
            ),
        ]
        assert_interactions(agent, interactions)
        assert agent.call_count == 2  # noqa: PLR2004

    def test_assert_interactions_fails_on_expectation(self):
        """Test assert_interactions when an expectation fails."""
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Wrong response")]])
        interactions = [
            Interaction(
                input="Hello",
                expectations=[MockFailingExpectation()],
            )
        ]
        with pytest.raises(AssertionError):
            assert_interactions(agent, interactions)

    def test_assert_interactions_log_response(self, caplog):
        """Test assert_interactions with log_response enabled."""
        logging.getLogger("llmqa.runners").setLevel(logging.INFO)
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Hello world")]])
        interactions = [
            Interaction(
                input="Hello",
                expectations=[MockPassingExpectation()],
            )
        ]
        assert_interactions(agent, interactions, log_response=True)
        assert "Response:" in caplog.text


class TestAbstractConversationalRunner:
    """Tests for AbstractConversationalRunner."""

    def test_abstract_runner_cannot_be_instantiated(self):
        """Test that AbstractConversationalRunner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractConversationalRunner()  # type: ignore[abstract]

    def test_abstract_runner_via_subclass(self):
        """Test AbstractConversationalRunner functionality via concrete subclass."""
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Hello world")]])
        runner = ConversationalAgentRunner(agent)
        # Test that it has the expected attributes
        assert runner.agent == agent
        assert runner.conversation_history == []


class TestConversationalRunner:
    """Tests for ConversationalRunner."""

    def test_conversational_runner_initialization(self):
        """Test ConversationalRunner initialization."""
        agent = MockAgent()
        runner = ConversationalAgentRunner(agent)
        assert runner.agent == agent
        assert runner.conversation_history == []

    def test_conversational_runner_single_interaction(self):
        """Test ConversationalRunner with a single interaction."""
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Hello world")]])
        runner = ConversationalAgentRunner(agent)
        interactions = [
            Interaction(
                input="Hello",
                expectations=[MockPassingExpectation()],
            )
        ]
        runner.assert_interactions(interactions)
        # Check conversation history
        assert len(runner.conversation_history) == 2  # noqa: PLR2004
        assert runner.conversation_history[0].content == "Hello"
        assert runner.conversation_history[1].content == "Hello world"

    def test_conversational_runner_multiple_interactions(self):
        """Test ConversationalRunner with multiple interactions maintaining history."""
        agent = MockAgent(
            responses=[
                [Message(role=ChatRole.ASSISTANT, content="Paris")],
                [Message(role=ChatRole.ASSISTANT, content="Eiffel Tower")],
            ]
        )
        runner = ConversationalAgentRunner(agent)
        interactions = [
            Interaction(
                input="What is the capital of France?",
                expectations=[MockPassingExpectation()],
            ),
            Interaction(
                input="What is its best known monument?",
                expectations=[MockPassingExpectation()],
            ),
        ]
        runner.assert_interactions(interactions)
        # Check conversation history has all messages
        assert len(runner.conversation_history) == 4  # noqa: PLR2004
        assert runner.conversation_history[0].content == "What is the capital of France?"
        assert runner.conversation_history[1].content == "Paris"
        assert runner.conversation_history[2].content == "What is its best known monument?"
        assert runner.conversation_history[3].content == "Eiffel Tower"

    def test_conversational_runner_conversation_history_persists(self):
        """Test that conversation history persists across multiple assert_interactions calls."""
        agent = MockAgent(
            responses=[
                [Message(role=ChatRole.ASSISTANT, content="First")],
                [Message(role=ChatRole.ASSISTANT, content="Second")],
            ]
        )
        runner = ConversationalAgentRunner(agent)
        # First call
        interactions1 = [
            Interaction(
                input="First question",
                expectations=[MockPassingExpectation()],
            )
        ]
        runner.assert_interactions(interactions1)
        assert len(runner.conversation_history) == 2  # noqa: PLR2004
        # Second call - history should persist
        interactions2 = [
            Interaction(
                input="Second question",
                expectations=[MockPassingExpectation()],
            )
        ]
        runner.assert_interactions(interactions2)
        assert len(runner.conversation_history) == 4  # noqa: PLR2004
        assert runner.conversation_history[0].content == "First question"
        assert runner.conversation_history[2].content == "Second question"

    def test_conversational_runner_full_history_sent_to_agent(self):
        """Test that full conversation history is sent to agent on each interaction."""
        agent = MockAgent(
            responses=[
                [Message(role=ChatRole.ASSISTANT, content="Response 1")],
                [Message(role=ChatRole.ASSISTANT, content="Response 2")],
            ]
        )
        runner = ConversationalAgentRunner(agent)
        interactions = [
            Interaction(
                input="First",
                expectations=[MockPassingExpectation()],
            ),
            Interaction(
                input="Second",
                expectations=[MockPassingExpectation()],
            ),
        ]
        runner.assert_interactions(interactions)
        # Check that agent received full history each time
        assert len(agent.received_messages) == 2  # noqa: PLR2004
        # First call: just the first input
        assert len(agent.received_messages[0]) == 1  # noqa: PLR2004
        assert agent.received_messages[0][0].content == "First"
        # Second call: first input + first response + second input
        assert len(agent.received_messages[1]) == 3  # noqa: PLR2004
        assert agent.received_messages[1][0].content == "First"
        assert agent.received_messages[1][1].content == "Response 1"
        assert agent.received_messages[1][2].content == "Second"

    def test_conversational_runner_fails_on_expectation(self):
        """Test ConversationalRunner when an expectation fails."""
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Wrong response")]])
        runner = ConversationalAgentRunner(agent)
        interactions = [
            Interaction(
                input="Hello",
                expectations=[MockFailingExpectation()],
            )
        ]
        with pytest.raises(AssertionError):
            runner.assert_interactions(interactions)

    def test_conversational_runner_log_conversation(self, caplog):
        """Test ConversationalRunner with log_conversation enabled."""
        logging.getLogger("llmqa.runners").setLevel(logging.INFO)
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Hello world")]])
        runner = ConversationalAgentRunner(agent)
        interactions = [
            Interaction(
                input="Hello",
                expectations=[MockPassingExpectation()],
            )
        ]
        runner.assert_interactions(interactions, log_conversation=True)
        assert "----------------------------------------" in caplog.text

    def test_conversational_runner_log_conversation_on_error(self, caplog):
        """Test ConversationalRunner logs conversation on error."""
        logging.getLogger("llmqa.runners").setLevel(logging.ERROR)
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Wrong response")]])
        runner = ConversationalAgentRunner(agent)
        interactions = [
            Interaction(
                input="Hello",
                expectations=[MockFailingExpectation()],
            )
        ]
        with pytest.raises(AssertionError):
            runner.assert_interactions(interactions, log_conversation_on_error=True)
        assert "-- Conversation History  -------------" in caplog.text

    def test_conversational_runner_build_log_conversation(self):
        """Test build_log_conversation method."""
        agent = MockAgent()
        runner = ConversationalAgentRunner(agent)
        runner.conversation_history = [
            Message(role=ChatRole.USER, content="Hello"),
            Message(role=ChatRole.ASSISTANT, content="Hi there"),
        ]
        log_str = runner.build_log_conversation()
        assert "-- Conversation History  -------------" in log_str
        assert "Hello" in log_str
        assert "Hi there" in log_str

    def test_conversational_runner_build_input_messages(self):
        """Test build_input_messages method."""
        agent = MockAgent()
        runner = ConversationalAgentRunner(agent)
        interaction = Interaction(
            input="Test message",
            expectations=[MockPassingExpectation()],
        )
        messages = runner.build_input_messages(interaction)
        assert len(messages) == 1  # noqa: PLR2004
        assert messages[0].content == "Test message"
        assert messages[0].role == ChatRole.USER

    def test_conversational_runner_with_message_input(self):
        """Test ConversationalRunner with Message input."""
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Response")]])
        runner = ConversationalAgentRunner(agent)
        message = Message(role=ChatRole.USER, content="Custom message")
        interactions = [
            Interaction(
                input=message,
                expectations=[MockPassingExpectation()],
            )
        ]
        runner.assert_interactions(interactions)
        assert runner.conversation_history[0] == message

    def test_conversational_runner_with_list_of_messages(self):
        """Test ConversationalRunner with list of messages as input."""
        agent = MockAgent(responses=[[Message(role=ChatRole.ASSISTANT, content="Response")]])
        runner = ConversationalAgentRunner(agent)
        messages = [
            Message(role=ChatRole.USER, content="Message 1"),
            Message(role=ChatRole.USER, content="Message 2"),
        ]
        interactions = [
            Interaction(
                input=messages,
                expectations=[MockPassingExpectation()],
            )
        ]
        runner.assert_interactions(interactions)
        # 2 input + 1 response
        assert len(runner.conversation_history) == 3  # noqa: PLR2004
        assert runner.conversation_history[0] == messages[0]
        assert runner.conversation_history[1] == messages[1]
