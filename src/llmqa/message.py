"""Lightweight Message class for LLM interactions.

This module defines a simplified Message class suitable for testing LLM responses
without the heavy dependencies of production message classes.
"""

from uuid import uuid4

from pydantic import BaseModel, Field

from .enums import ChatRole, MessageType


class Message(BaseModel):
    """Represents a single message in a chat-like system.

    This is a lightweight version designed for testing purposes, without
    MongoDB-specific fields or complex token tracking.

    Attributes:
        role: The role of the message sender (e.g., user, assistant).
        content: The textual content of the message.
        type: The type of the message (default is TEXT).
        id: Unique identifier for the message.

    Example:
        Creating a new message:

            from llmqa.message import Message
            from llmqa.enums import ChatRole

            msg = Message(
                role=ChatRole.USER,
                content="Hello, how can I assist you today?"
            )
    """

    role: ChatRole
    content: str
    type: MessageType = MessageType.TEXT
    id: str = Field(default_factory=lambda: str(uuid4()))

    def __str__(self) -> str:
        """Return the content of the message."""
        return self.content

    def __repr__(self) -> str:
        """Return a detailed string representation of the message."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message(role={self.role}, content='{content_preview}', type={self.type})"
