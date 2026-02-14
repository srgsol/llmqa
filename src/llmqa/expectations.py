"""Expectation handlers for validating LLM responses.

This module contains a set of predefined expectation classes for testing
LLM agent responses.

Each expectation class inherits from BaseExpectation and implements the _run
method to perform validation logic.

Example:
    >>> from llmqa.expectations import TextContains
    >>> from llmqa.message import Message
    >>> from llmqa.enums import ChatRole
    >>> expectation = TextContains(text="Hello", msg_id=0)
    >>> response = [Message(role=ChatRole.ASSISTANT, content="Hello world!")]
    >>> expectation.run(response)  # Passes silently

"""

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from .enums import MessageType
from .function_call import FunctionCall, extract_function_calls
from .message import Message


class BaseExpectation(BaseModel, ABC):
    """Base class for all expectation validators.

    Expectations are used to validate agent responses in tests. Subclasses
    should implement the `_run` method to perform the actual validation logic.

    Attributes:
        assert_raises: Optional exception type to expect. If set, the expectation
            passes only if `_run` raises an exception of this type. If None
            (default), exceptions from `_run` are re-raised.

    Example:
        Creating a custom expectation:

            class TextLength(BaseExpectation):
                min_length: int
                msg_id: int = 0

                def _run(self, response: list[Message]) -> None:
                    actual = len(response[self.msg_id].content)
                    if actual < self.min_length:
                        raise AssertionError(f"Too short: {actual} < {self.min_length}")
    """

    assert_raises: type[Exception] | None = None

    def run(self, response: list[Message]) -> None:
        """Execute the expectation validation against the response.

        This method wraps the concrete `_run` implementation with error handling.
        If `assert_raises` is None, any exception from `_run` will be re-raised.
        If `assert_raises` is set, only exceptions of that type are caught and
        the expectation passes; other exceptions are re-raised.

        Args:
            response: List of Message objects from the agent to validate against.

        Raises:
            Exception: If the expectation fails.
        """
        try:
            self._run(response)
        except Exception as e:
            # Prefix all exceptions messages with the concrete Expectation class name.
            class_name = self.__class__.__name__
            original_msg = str(e) if e.args else ""
            prefixed_msg = f"({class_name}) {original_msg}" if original_msg else class_name
            # Modify the exception message in place
            if e.args:
                e.args = (prefixed_msg,) + e.args[1:]
            else:
                e.args = (prefixed_msg,)
            if self.assert_raises is None:
                raise e
            if not isinstance(e, self.assert_raises):
                raise e
            return

    @abstractmethod
    def _run(self, response: list[Message]) -> None:
        """Implement the concrete expectation validation logic.

        Subclasses must implement this method to perform the actual validation.
        This method should raise an AssertionError if the expectation is not met.

        Args:
            response: List of Message objects from the agent to validate against.

        Raises:
            AssertionError: If the expectation is not met.
        """
        ...


class IsJson(BaseExpectation):
    """Check that the message content is valid JSON."""

    msg_id: int = 0

    def _run(self, response: list[Message]) -> None:
        """Check the message content is valid JSON."""
        content = response[self.msg_id].content
        try:
            json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            raise AssertionError(f"Message {self.msg_id} is not valid JSON: {e}") from e


class NumMessages(BaseExpectation):
    """Check that the number of messages matches the expected count."""

    num: int

    def _run(self, response: list[Message]) -> None:
        """Check the number of messages matches."""
        actual_num = len(response)
        assertion = actual_num == self.num
        err_msg = f"Expected {self.num} messages but got {actual_num}"
        assert assertion, err_msg


class TextContains(BaseExpectation):
    """Check that the text is in the response."""

    msg_id: int = 0
    text: str
    case_sensitive: bool = True

    def _run(self, response: list[Message]) -> None:
        """Check the text is in the response."""
        actual_output = response[self.msg_id].content
        search_text = self.text
        if not self.case_sensitive:
            actual_output = actual_output.lower()
            search_text = search_text.lower()

        assertion = search_text in actual_output
        err_msg = (
            f"Expected text '{self.text}' (case-sensitive: {self.case_sensitive}) "
            f"not found in Message-{self.msg_id}: '{actual_output}'"
        )
        assert assertion, err_msg


class TextContainsAll(BaseExpectation):
    """Check that all texts are in the response."""

    msg_id: int = 0
    text: list[str]
    case_sensitive: bool = True

    def _run(self, response: list[Message]) -> None:
        """Check all texts are in the response."""
        actual_output = response[self.msg_id].content
        search_texts = self.text
        if not self.case_sensitive:
            actual_output = actual_output.lower()
            search_texts = [t.lower() for t in self.text]

        missing = [
            t for t, st in zip(self.text, search_texts, strict=False) if st not in actual_output
        ]
        assertion = len(missing) == 0
        err_msg = (
            f"Expected texts {missing} (case-sensitive: {self.case_sensitive}) "
            f"not found in Message-{self.msg_id}: '{actual_output}'"
        )
        assert assertion, err_msg


class TextContainsAny(BaseExpectation):
    """Check that any of the texts is in the response."""

    msg_id: int = 0
    text: list[str]
    case_sensitive: bool = True

    def _run(self, response: list[Message]) -> None:
        """Check any of the texts is in the response."""
        actual_output = response[self.msg_id].content
        search_texts = self.text
        if not self.case_sensitive:
            actual_output = actual_output.lower()
            search_texts = [t.lower() for t in self.text]

        found = any(text_item in actual_output for text_item in search_texts)
        err_msg = (
            f"None of the expected texts {self.text} (case-sensitive: {self.case_sensitive}) "
            f"were found in Message-{self.msg_id}: '{actual_output}'"
        )
        assert found, err_msg


class TextExactMatch(BaseExpectation):
    """Check that the text exactly matches the response."""

    msg_id: int = 0
    text: str
    case_sensitive: bool = True

    def _run(self, response: list[Message]) -> None:
        """Check the text exactly matches the response."""
        actual_output = response[self.msg_id].content
        expected_text = self.text
        if not self.case_sensitive:
            actual_output = actual_output.lower()
            expected_text = expected_text.lower()

        assertion = actual_output == expected_text
        err_msg = (
            f"Expected exact text '{self.text}' (case-sensitive: {self.case_sensitive}) "
            f"not found in Message-{self.msg_id}: '{actual_output}'"
        )
        assert assertion, err_msg


class TextIsEmpty(BaseExpectation):
    """Check that the message content is empty."""

    msg_id: int = 0

    def _run(self, response: list[Message]) -> None:
        """Check the message content is empty."""
        actual_output = response[self.msg_id].content
        assertion = len(actual_output) == 0
        err_msg = f"Message {self.msg_id} is not empty: <{actual_output}>"
        assert assertion, err_msg


class TextIsInRange(BaseExpectation):
    """Check that the word count is in the specified range."""

    msg_id: int = 0
    min_words: int
    max_words: int

    def _run(self, response: list[Message]) -> None:
        """Check the word count is in range."""
        actual_output = response[self.msg_id].content
        word_count = len(actual_output.split())
        assertion = self.min_words <= word_count <= self.max_words
        err_msg = (
            f"Message-{self.msg_id} has {word_count} words, which is not in range "
            f"[{self.min_words}, {self.max_words}]"
        )
        assert assertion, err_msg


class TextIsNotEmpty(BaseExpectation):
    """Check that the message content is not empty."""

    msg_id: int = 0

    def _run(self, response: list[Message]) -> None:
        """Check the message content is not empty."""
        actual_output = response[self.msg_id].content
        assertion = len(actual_output) > 0
        err_msg = f"Message-{self.msg_id} is empty"
        assert assertion, err_msg


class TextNotContains(BaseExpectation):
    """Check that the text is not in the response."""

    msg_id: int = 0
    text: str
    case_sensitive: bool = True

    def _run(self, response: list[Message]) -> None:
        """Check the text is not in the response."""
        actual_output = response[self.msg_id].content
        search_text = self.text
        if not self.case_sensitive:
            actual_output = actual_output.lower()
            search_text = search_text.lower()

        assertion = search_text not in actual_output
        err_msg = (
            f"Text '{self.text}' (case-sensitive: {self.case_sensitive}) "
            f"should not be found in Message-{self.msg_id}, but it was found"
        )
        assert assertion, err_msg


class TextNotContainsAny(BaseExpectation):
    """Check that none of the texts are in the response."""

    msg_id: int = 0
    text: list[str]
    case_sensitive: bool = True

    def _run(self, response: list[Message]) -> None:
        """Check none of the texts are in the response."""
        actual_output = response[self.msg_id].content
        search_texts = self.text
        if not self.case_sensitive:
            actual_output = actual_output.lower()
            search_texts = [t.lower() for t in self.text]

        found_any = any(text_item in actual_output for text_item in search_texts)
        assertion = not found_any
        err_msg = (
            f"Some of the texts {self.text} (case-sensitive: {self.case_sensitive}) "
            f"were found in Message-{self.msg_id}, but none should be present"
        )
        assert assertion, err_msg


class TextNotStartsWith(BaseExpectation):
    """Check that the message does not start with the text."""

    msg_id: int = 0
    text: str
    case_sensitive: bool = True

    def _run(self, response: list[Message]) -> None:
        """Check the message does not start with the text."""
        actual_output = response[self.msg_id].content
        prefix = self.text
        if not self.case_sensitive:
            actual_output = actual_output.lower()
            prefix = prefix.lower()

        assertion = not actual_output.startswith(prefix)
        err_msg = (
            f"Message-{self.msg_id}: '{actual_output}' must not start with "
            f"'{self.text}' (case-sensitive: {self.case_sensitive})"
        )
        assert assertion, err_msg


class TextStartsWith(BaseExpectation):
    """Check that the message starts with the text."""

    msg_id: int = 0
    text: str
    case_sensitive: bool = True

    def _run(self, response: list[Message]) -> None:
        """Check the message starts with the text."""
        actual_output = response[self.msg_id].content
        prefix = self.text
        if not self.case_sensitive:
            actual_output = actual_output.lower()
            prefix = prefix.lower()

        assertion = actual_output.startswith(prefix)
        err_msg = (
            f"Message-{self.msg_id}: '{actual_output}' must start with "
            f"'{self.text}' (case-sensitive: {self.case_sensitive})"
        )
        assert assertion, err_msg


class ToolCallOutput(BaseExpectation):
    """Check that the message is a tool call output."""

    msg_id: int = 0

    def _run(self, response: list[Message]) -> None:
        """Check the message is a tool call output."""
        message = response[self.msg_id]
        assertion = message.type == MessageType.FUNCTION_OUTPUT
        err_msg = f"Message-{self.msg_id} is not a tool call output"
        assert assertion, err_msg


class ToolCallRequest(BaseExpectation):
    """Check that the message is a tool call request with optional tool name and parameters."""

    msg_id: int = 0
    tool_name: str | None = None
    tool_parameters: dict[str, Any] | None = None

    def _run(self, response: list[Message]) -> None:
        """Check the message is a tool call request."""
        message = response[self.msg_id]

        # Check message type
        assertion = message.type == MessageType.FUNCTION_REQUEST
        err_msg = f"Message-{self.msg_id} is not a tool call request"
        assert assertion, err_msg

        # Check correctness of expectation definition
        if self.tool_parameters is not None and self.tool_name is None:
            raise ValueError("Tool name is required when tool parameters are provided")

        # If no tool name is specified, just check that it's a function request
        if self.tool_name is None:
            return

        # Parse the message to get the function calls
        try:
            function_calls: list[FunctionCall] = extract_function_calls(message)

            if not function_calls:
                raise AssertionError(f"No tool calls found in Message-{self.msg_id}")

            # Find the function call with the matching tool name
            function_call = None
            for func_call in function_calls:
                if func_call.name == self.tool_name:
                    function_call = func_call
                    break

            if function_call is None:
                raise AssertionError(
                    f"Function call for tool '{self.tool_name}' not found in Message-{self.msg_id}"
                )

            # Check if the tool name is correct
            assertion = function_call.name == self.tool_name
            err_msg = (
                f" Incorrect tool name. Expected '{self.tool_name}' but got "
                f"'{function_call.name}' in Message-{self.msg_id}"
            )
            assert assertion, err_msg

            # Check if the tool parameters are correct
            if self.tool_parameters is not None:
                for key, value in self.tool_parameters.items():
                    assertion = key in function_call.parameters
                    err_msg = (
                        f"Tool parameter '{key}' not found in function call in "
                        f"Message-{self.msg_id}"
                    )
                    assert assertion, err_msg

                    assertion = function_call.parameters[key] == value
                    err_msg = (
                        f"Tool parameter '{key}': Expected {value} but got "
                        f"{function_call.parameters[key]} in Message-{self.msg_id}"
                    )
                    assert assertion, err_msg
        except AssertionError:
            raise
        except Exception as e:
            raise AssertionError(f"Error parsing the message: {e}") from e
