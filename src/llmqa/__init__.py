"""LLMQA - A testing framework for LLM prompts and agents.

This package provides tools for testing LLM responses through expectations
and test runners.

Example:
    Basic usage:

        from llmqa import AgentQA, ConversationalAgentRunner, TextContains

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

from .agent import Agent, AgentQA
from .enums import ChatRole, MessageType
from .error import (
    AgentError,
    AgentGenerationError,
    AuthenticationError,
    FunctionCallParsingError,
    LLMQAError,
    ProviderError,
    RateLimitError,
)
from .expectations import (
    BaseExpectation,
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
from .function_call import FunctionCall, extract_function_calls
from .llm_model import LLMModel
from .message import Message
from .provider import LLMProvider
from .runners import (
    AbstractConversationalRunner,
    ConversationalAgentRunner,
    Interaction,
    assert_expectations,
    assert_interactions,
    build_input_messages,
)
from .utils import render_conversation

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Agent",
    "AgentQA",
    "Message",
    "LLMModel",
    "LLMProvider",
    # Enums
    "ChatRole",
    "MessageType",
    # Expectations
    "BaseExpectation",
    "IsJson",
    "NumMessages",
    "TextContains",
    "TextContainsAll",
    "TextContainsAny",
    "TextExactMatch",
    "TextIsEmpty",
    "TextIsInRange",
    "TextIsNotEmpty",
    "TextNotContains",
    "TextNotContainsAny",
    "TextNotStartsWith",
    "TextStartsWith",
    "ToolCallOutput",
    "ToolCallRequest",
    # Runners
    "AbstractConversationalRunner",
    "ConversationalAgentRunner",
    "Interaction",
    "assert_expectations",
    "assert_interactions",
    "build_input_messages",
    # Function calls
    "FunctionCall",
    "extract_function_calls",
    # Utilities
    "render_conversation",
    # Errors
    "LLMQAError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "AgentError",
    "AgentGenerationError",
    "FunctionCallParsingError",
]
