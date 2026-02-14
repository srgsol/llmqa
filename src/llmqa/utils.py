"""Utility functions for the llmqa testing framework.

This module provides helper functions used across the testing framework.
"""

from .message import Message


def render_conversation(conversation: list[Message]) -> str:
    """Render a conversation as a formatted string.

    This is useful for logging and debugging test failures, providing
    a readable view of the conversation history.

    Args:
        conversation: List of Message objects representing the conversation.

    Returns:
        A formatted string representation of the conversation.

    Example:
        >>> from llmqa.message import Message
        >>> from llmqa.enums import ChatRole
        >>> msgs = [
        ...     Message(role=ChatRole.USER, content="Hello"),
        ...     Message(role=ChatRole.ASSISTANT, content="Hi there!"),
        ... ]
        >>> print(render_conversation(msgs))
        -- Conversation History  -------------
            USER: Hello
        ----------------------------------------
            ASSISTANT: Hi there!
        ----------------------------------------
    """
    conversation_str = "\n-- Conversation History  -------------\n"
    for msg in conversation:
        conversation_str += f"\t{msg.role.value}: {msg.content}\n"
        conversation_str += "----------------------------------------\n"
    return conversation_str
