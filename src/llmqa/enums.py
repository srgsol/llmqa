"""Core enumerations for the llmqa testing framework.

This module defines the foundational enums used across the system for
message handling and chat interactions.
"""

from enum import Enum


class ChatRole(str, Enum):
    """Defines participant roles in chat interactions.

    Attributes:
        ASSISTANT: Bot/AI responses
        USER: Human inputs
        SYSTEM: System-level instructions
        FUNCTION: Function call related messages
    """

    ASSISTANT = "ASSISTANT"
    USER = "USER"
    SYSTEM = "SYSTEM"
    FUNCTION = "FUNCTION"


class MessageType(str, Enum):
    """Categorizes different types of messages in the system.

    Attributes:
        TEXT: Standard text messages
        FUNCTION_REQUEST: Function invocation requests
        FUNCTION_OUTPUT: Results from function executions
    """

    TEXT = "TEXT"
    FUNCTION_REQUEST = "FUNCTION_REQUEST"
    FUNCTION_OUTPUT = "FUNCTION_OUTPUT"
