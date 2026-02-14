# LLMQA

A testing framework for LLM prompts and agents.

## Overview

LLMQA provides tools for testing LLM responses through expectations and test runners. It allows you to write declarative tests for your LLM-powered applications.

## Installation

```bash
pip install llmqa
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```python
from llmqa import AgentQA, ConversationalAgentRunner, TextContains

# Create a QA agent (uses OpenAI by default)
agent = AgentQA(model="gpt-4o")

# Create a runner that maintains conversation history
runner = ConversationalAgentRunner(agent)

# Define interactions with expectations
interactions = [
    {
        "input": "What is the capital of France?",
        "expectations": [TextContains(text="Paris")],
    },
    {
        "input": "What is its best known monument?",
        "expectations": [TextContains(text="Eiffel")],
    },
]

# Run the test
runner.assert_interactions(interactions)
```

## Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Or pass it directly:

```python
agent = AgentQA(model="gpt-4o", api_key="your-api-key")
```

## Core Concepts

### Expectations

Expectations validate agent responses. Each expectation inherits from `BaseExpectation` and implements validation logic.

**Available Expectations:**

| Expectation | Description |
|-------------|-------------|
| `TextContains` | Check if text is in the response |
| `TextContainsAll` | Check if all texts are in the response |
| `TextContainsAny` | Check if any text is in the response |
| `TextExactMatch` | Check if text exactly matches |
| `TextStartsWith` | Check if response starts with text |
| `TextNotContains` | Check if text is NOT in the response |
| `TextNotContainsAny` | Check if none of the texts are in response |
| `TextNotStartsWith` | Check if response does NOT start with text |
| `TextIsEmpty` | Check if response is empty |
| `TextIsNotEmpty` | Check if response is not empty |
| `TextIsInRange` | Check if word count is in range |
| `NumMessages` | Check number of messages |
| `IsJson` | Check if response is valid JSON |
| `ToolCallRequest` | Check for tool call requests |
| `ToolCallOutput` | Check for tool call outputs |

### Interactions

An `Interaction` represents a single test case:
- **input**: User message(s) to send to the agent
- **expectations**: List of expectations to validate the response

```python
from llmqa import Interaction, TextContains

interaction = Interaction(
    input="What is Python?",
    expectations=[TextContains(text="programming")],
)
```

### Runners

Three ways to run tests:

1. **Standalone** (`assert_interactions`): Each interaction is independent
2. **Conversational** (`ConversationalAgentRunner`): Maintains conversation history
3. **Direct** (`assert_expectations`): Validate expectations against existing response

## Usage Examples

### Basic Test

```python
from llmqa import AgentQA, assert_interactions, TextContains

agent = AgentQA()

interactions = [
    {
        "input": "What is the capital of France?",
        "expectations": [TextContains(text="Paris", msg_id=0)],
    }
]

assert_interactions(agent, interactions)
```

### Conversational Test

```python
from llmqa import AgentQA, ConversationalAgentRunner, TextContains

agent = AgentQA()
runner = ConversationalAgentRunner(agent)

interactions = [
    {
        "input": "What is the capital of France?",
        "expectations": [TextContains(text="Paris")],
    },
    {
        "input": "What is its best known monument?",
        "expectations": [TextContains(text="Eiffel Tower")],
    },
]

runner.assert_interactions(interactions, log_conversation=True)
```

### Multiple Expectations

```python
from llmqa import TextContains, TextNotContains, NumMessages

interactions = [
    {
        "input": "Explain quantum computing",
        "expectations": [
            NumMessages(num=1),
            TextContains(text="quantum"),
            TextNotContains(text="classical"),
        ],
    }
]
```

### Direct Expectations

```python
from llmqa import Message, ChatRole, assert_expectations, TextContains, NumMessages

response = [
    Message(role=ChatRole.ASSISTANT, content="Python is a programming language"),
]

expectations = [
    NumMessages(num=1),
    TextContains(text="Python"),
]

assert_expectations(expectations, response)
```

## Creating Custom Expectations

```python
from llmqa import BaseExpectation, Message

class TextLength(BaseExpectation):
    """Check that the message content has a minimum length."""

    msg_id: int = 0
    min_length: int

    def _run(self, response: list[Message]) -> None:
        actual_length = len(response[self.msg_id].content)
        if actual_length < self.min_length:
            raise AssertionError(
                f"Message {self.msg_id} has length {actual_length}, "
                f"but minimum required is {self.min_length}"
            )
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
ruff format .
```

Run type checking:

```bash
mypy llmqa
```

## License

MIT License
