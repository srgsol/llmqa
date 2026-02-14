"""Parametrized tests for expectation classes."""

import pytest

from llmqa.enums import ChatRole, MessageType
from llmqa.expectations import (
    IsJson,
    NumMessages,
    TextContains,
    TextContainsAll,
    TextContainsAny,
    TextExactMatch,
    TextIsEmpty,
    TextIsInRange,
    TextIsNotEmpty,
    TextNotContains,
    TextNotContainsAny,
    TextNotStartsWith,
    TextStartsWith,
    ToolCallOutput,
    ToolCallRequest,
)
from llmqa.message import Message


@pytest.mark.parametrize(
    ("content", "should_pass"),
    [
        ('{"key": "value"}', True),
        ('{"name": "John", "age": 30}', True),
        ("not json", False),
        ('{"incomplete": json', False),
        ('Once upon a time John {"name": "John", "age": 30}', False),
    ],
)
def test_is_json(content: str, should_pass: bool):
    """Test IsJson expectation."""
    expectation = IsJson()
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]
    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("messages_count", "expected_num", "should_pass"),
    [
        (1, 1, True),
        (2, 2, True),
        (0, 0, True),
        (1, 2, False),
        (2, 1, False),
    ],
)
def test_num_messages(messages_count: int, expected_num: int, should_pass: bool):
    """Test NumMessages expectation."""
    expectation = NumMessages(num=expected_num)
    messages = [Message(role=ChatRole.ASSISTANT, content=f"msg{i}") for i in range(messages_count)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "text", "case_sensitive", "should_pass"),
    [
        ("Hello World", "World", True, True),
        ("Hello World", "Hello", True, True),
        ("Hello World", "Python", True, False),
        ("Hello World", "world", True, False),
        ("Hello World", "world", False, True),
        ("Hello World", "", True, True),
        ("Hello Beautiful World", "Beautiful", True, True),
        ("Hello Hello World", "Hello", True, True),
        ("", "Hello", True, False),
    ],
)
def test_text_contains(content: str, text: str, case_sensitive: bool, should_pass: bool):
    """Test TextContains expectation."""
    expectation = TextContains(text=text, case_sensitive=case_sensitive)
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "texts", "case_sensitive", "should_pass"),
    [
        ("Hello World", ["Hello", "World"], True, True),
        ("Hello World", ["Hello", "Python"], True, False),
        ("Hello World", ["hello", "world"], True, False),
        ("Hello World", ["hello", "world"], False, True),
        ("Hello Beautiful World", ["Hello", "Beautiful", "World"], True, True),
        ("Hello Beautiful World", ["Hello", "World"], True, True),
        ("Hello World", [], True, True),
        ("Hello World", [""], True, True),
        ("", ["Hello", "World"], True, False),
    ],
)
def test_text_contains_all(content: str, texts: list[str], case_sensitive: bool, should_pass: bool):
    """Test TextContainsAll expectation."""
    expectation = TextContainsAll(text=texts, case_sensitive=case_sensitive)
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "texts", "case_sensitive", "should_pass"),
    [
        ("Hello World", ["World", "Python"], True, True),
        ("Hello World", ["Python", "Java"], True, False),
        ("Hello World", ["world", "python"], True, False),
        ("Hello World", ["world", "python"], False, True),
        ("Hello World", ["Hello", "World"], True, True),
        ("Hello World", [], True, False),
        ("Hello World", ["", "Hello"], True, True),
        ("", ["Hello", "World"], True, False),
        ("Hello Beautiful World", ["Beautiful", "Python"], True, True),
    ],
)
def test_text_contains_any(content: str, texts: list[str], case_sensitive: bool, should_pass: bool):
    """Test TextContainsAny expectation."""
    expectation = TextContainsAny(text=texts, case_sensitive=case_sensitive)
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "text", "case_sensitive", "should_pass"),
    [
        ("Hello World", "Hello World", True, True),
        ("Hello World", "Hello", True, False),
        ("Hello World", "hello world", True, False),
        ("Hello World", "hello world", False, True),
        ("", "", True, True),
    ],
)
def test_text_exact_match(content: str, text: str, case_sensitive: bool, should_pass: bool):
    """Test TextExactMatch expectation."""
    expectation = TextExactMatch(text=text, case_sensitive=case_sensitive)
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "should_pass"),
    [
        ("", True),
        ("Hello World", False),
        (" ", False),
        ("\n\t", False),
        ("Hello", False),
    ],
)
def test_text_is_empty(content: str, should_pass: bool):
    """Test TextIsEmpty expectation."""
    expectation = TextIsEmpty()
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "min_words", "max_words", "should_pass"),
    [
        ("Hello World", 1, 3, True),
        ("Hello World", 2, 2, True),
        ("Hello World", 1, 1, False),
        ("Hello World", 3, 5, False),
        ("One two three four five", 3, 5, True),
        ("One two three four", 3, 5, True),
        ("One two three", 3, 5, True),
        ("One two", 3, 5, False),
        ("One two three four five", 1, 2, False),
        ("One two three four five", 5, 10, True),
        ("", 0, 0, True),
        ("", 1, 5, False),
        ("Single", 1, 1, True),
        ("Single", 2, 5, False),
        ("This is a longer sentence with many words", 5, 10, True),
        ("This is a longer sentence with many words", 1, 4, False),
    ],
)
def test_text_is_in_range(content: str, min_words: int, max_words: int, should_pass: bool):
    """Test TextIsInRange expectation."""
    expectation = TextIsInRange(min_words=min_words, max_words=max_words)
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "should_pass"),
    [
        ("", False),
        ("Hello World", True),
        (" ", True),
        ("\n\t", True),
        ("Hello", True),
    ],
)
def test_text_is_not_empty(content: str, should_pass: bool):
    """Test TextIsNotEmpty expectation."""
    expectation = TextIsNotEmpty()
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "text", "case_sensitive", "should_pass"),
    [
        ("Hello World", "Python", True, True),
        ("Hello World", "World", True, False),
        ("Hello World", "world", True, True),
        ("Hello World", "world", False, False),
        ("Hello World", "Hello", True, False),
        ("Hello World", "", True, False),
        ("", "Hello", True, True),
        ("Hello Beautiful World", "Python", True, True),
    ],
)
def test_text_not_contains(content: str, text: str, case_sensitive: bool, should_pass: bool):
    """Test TextNotContains expectation."""
    expectation = TextNotContains(text=text, case_sensitive=case_sensitive)
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "texts", "case_sensitive", "should_pass"),
    [
        ("Hello World", ["Python", "Java"], True, True),
        ("Hello World", ["World", "Python"], True, False),
        ("Hello World", ["world", "python"], True, True),
        ("Hello World", ["world", "python"], False, False),
        ("Hello World", ["Hello", "World"], True, False),
        ("Hello World", [], True, True),
        ("", ["Hello", "World"], True, True),
        ("Hello Beautiful World", ["Python", "Java"], True, True),
    ],
)
def test_text_not_contains_any(
    content: str, texts: list[str], case_sensitive: bool, should_pass: bool
):
    """Test TextNotContainsAny expectation."""
    expectation = TextNotContainsAny(text=texts, case_sensitive=case_sensitive)
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "text", "case_sensitive", "should_pass"),
    [
        ("Hello World", "World", True, True),
        ("Hello World", "Hello", True, False),
        ("Hello World", "hello", True, True),
        ("Hello World", "hello", False, False),
        ("Hello World", "Python", True, True),
        ("Hello World", "", True, False),
        ("", "Hello", True, True),
        ("Hello Beautiful World", "World", True, True),
    ],
)
def test_text_not_startswith(content: str, text: str, case_sensitive: bool, should_pass: bool):
    """Test TextNotStartswith expectation."""
    expectation = TextNotStartsWith(text=text, case_sensitive=case_sensitive)
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("content", "text", "case_sensitive", "should_pass"),
    [
        ("Hello World", "Hello", True, True),
        ("Hello World", "Hello World", True, True),
        ("Hello World", "World", True, False),
        ("Hello World", "hello", True, False),
        ("Hello World", "hello", False, True),
        ("Hello World", "", True, True),
        ("", "Hello", True, False),
        ("", "", True, True),
        ("Hello Beautiful World", "Hello", True, True),
    ],
)
def test_text_startswith(content: str, text: str, case_sensitive: bool, should_pass: bool):
    """Test TextStartswith expectation."""
    expectation = TextStartsWith(text=text, case_sensitive=case_sensitive)
    messages = [Message(role=ChatRole.ASSISTANT, content=content)]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("message_type", "should_pass"),
    [
        (MessageType.FUNCTION_OUTPUT, True),
        (MessageType.TEXT, False),
        (MessageType.FUNCTION_REQUEST, False),
    ],
)
def test_tool_call_output(message_type: MessageType, should_pass: bool):
    """Test ToolCallOutput expectation."""
    expectation = ToolCallOutput()
    message = Message(role=ChatRole.ASSISTANT, type=message_type, content="result")
    messages = [message]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)


@pytest.mark.parametrize(
    ("message_type", "should_pass"),
    [
        (MessageType.FUNCTION_REQUEST, True),
        (MessageType.TEXT, False),
        (MessageType.FUNCTION_OUTPUT, False),
    ],
)
def test_tool_call_request(message_type: MessageType, should_pass: bool):
    """Test ToolCallRequest expectation."""
    expectation = ToolCallRequest()
    message = Message(
        role=ChatRole.ASSISTANT,
        type=message_type,
        content='{"type": "function", "name": "test", "parameters": {}}',
    )
    messages = [message]

    if should_pass:
        expectation.run(messages)
    else:
        with pytest.raises(AssertionError):
            expectation.run(messages)
