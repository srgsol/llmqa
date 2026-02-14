"""Function call parsing utilities.

This module provides utilities for parsing function calls from LLM responses.
It extracts the JSON parsing logic needed for validating tool call expectations.

Example:
    Parsing function calls from a message:

        from llmqa.function_call import extract_function_calls
        from llmqa.message import Message
        from llmqa.enums import ChatRole

        message = Message(
            role=ChatRole.ASSISTANT,
            content='{"type": "function", "name": "search", "parameters": {"query": "test"}}'
        )
        function_calls = extract_function_calls(message)
        print(function_calls[0].name)  # "search"
"""

import json
from dataclasses import dataclass
from typing import Any

from .error import FunctionCallParsingError
from .message import Message


@dataclass
class FunctionCall:
    """Represents a parsed function call from LLM output.

    Attributes:
        call_string: Raw function call string from LLM (JSON representation).
        name: Name of the function to call.
        parameters: Dictionary of parameters for the function.
    """

    call_string: str
    name: str
    parameters: dict[str, Any]

    def __eq__(self, other: object) -> bool:
        """Check if two function calls are equal."""
        if isinstance(other, FunctionCall):
            return self.name == other.name and self.parameters == other.parameters
        return False

    def __repr__(self) -> str:
        """Return the string representation of the function call."""
        return self.call_string

    def __str__(self) -> str:
        """Return the string representation of the function call."""
        return self.call_string

    def __hash__(self) -> int:
        """Return the hash of the function call."""
        parameters_json = json.dumps(self.parameters, sort_keys=True)
        return hash((self.name, parameters_json))


def extract_function_calls(message: Message) -> list[FunctionCall]:
    """Extract function calls from a message.

    Parses the message content to find JSON objects that represent function calls.
    A function call is identified by having a "type": "function" field.

    Args:
        message: The message to parse for function calls.

    Returns:
        List of FunctionCall objects found in the message.

    Example:
        >>> msg = Message(
        ...     role=ChatRole.ASSISTANT,
        ...     content='{"type": "function", "name": "get_weather", "parameters": {"city": "NYC"}}'
        ... )
        >>> calls = extract_function_calls(msg)
        >>> calls[0].name
        'get_weather'
    """
    json_objects = _extract_json(message.content)
    function_call_objects = [
        obj for obj in json_objects if obj.get("type") == "function"
    ]
    return _build_function_calls(function_call_objects)


def _extract_json(text: str) -> list[dict[str, Any]]:
    """Extract JSON objects from text.

    Scans through the text and extracts all valid JSON objects and arrays.
    For arrays, each dict item is extracted individually.

    Args:
        text: The text to parse for JSON objects.

    Returns:
        List of dictionaries found in the text.
    """
    json_objects: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = 0

    while idx < len(text):
        try:
            obj, idx_end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                json_objects.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        json_objects.append(item)
            idx = idx_end
        except json.JSONDecodeError:
            idx += 1  # Skip one character if no valid JSON found

    return json_objects


def _build_function_calls(
    function_call_objects: list[dict[str, Any]],
) -> list[FunctionCall]:
    """Build FunctionCall objects from parsed JSON.

    Args:
        function_call_objects: List of dictionaries representing function calls.

    Returns:
        List of FunctionCall objects.

    Raises:
        FunctionCallParsingError: If a function call is malformed.
    """
    funcs: list[FunctionCall] = []

    for item in function_call_objects:
        try:
            function_name = item["name"]
            function_parameters = item.get("parameters", {})

            function_call = FunctionCall(
                call_string=json.dumps(item),
                name=function_name,
                parameters=function_parameters,
            )

            if function_call not in funcs:
                funcs.append(function_call)

        except KeyError as e:
            raise FunctionCallParsingError(
                f"Missing required key {e} in function call: {item}"
            ) from e
        except TypeError as e:
            raise FunctionCallParsingError(
                f"Invalid function call format: {item}"
            ) from e

    return funcs


def parse_function_call_strict(
    message: Message,
    expected_tool_name: str | None = None,
) -> FunctionCall | None:
    """Parse a single function call with optional name validation.

    This is a convenience function for cases where you expect exactly one
    function call and want to validate its name.

    Args:
        message: The message to parse.
        expected_tool_name: If provided, validates that the function call
            matches this name.

    Returns:
        The FunctionCall if found and valid, None otherwise.

    Raises:
        FunctionCallParsingError: If expected_tool_name is provided but
            no matching function call is found.
    """
    function_calls = extract_function_calls(message)

    if not function_calls:
        if expected_tool_name:
            raise FunctionCallParsingError(
                f"Expected function call '{expected_tool_name}' but none found"
            )
        return None

    if expected_tool_name:
        for fc in function_calls:
            if fc.name == expected_tool_name:
                return fc
        raise FunctionCallParsingError(
            f"Expected function call '{expected_tool_name}' but found: "
            f"{[fc.name for fc in function_calls]}"
        )

    return function_calls[0]
