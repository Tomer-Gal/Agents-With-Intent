# Agents-With-Intent

A Python library for building LangGraph-based AI agents with progressive skill discovery following the official [Agent Skills specification](https://agentskills.io/).

## Features

- 🎯 **LangGraph Integration** - Built on LangGraph state machines for robust agent orchestration
- 📚 **Progressive Skill Discovery** - Load skill metadata upfront, full content on-demand
- 🔧 **Agent Skills Spec Compliant** - Follows official specification from agentskills.io
- 🔌 **LLM Agnostic** - Use any LangChain-compatible LLM (OpenAI, Anthropic, Ollama, etc.)
- 🚀 **Script Execution** - Run executable scripts from skill directories
- 📖 **Reference Loading** - Access detailed documentation when needed
- 🛠️ **Tool Support** - Bind and execute LangChain tools

## Installation

```bash
pip install agents-with-intent
```

Or install from source:

```bash
git clone https://github.com/Tomer-Gal/Agents-With-Intent.git
cd Agents-With-Intent
pip install -e .
```

## Quick Start

```python
from agents_with_intent import Agent
from langchain_openai import ChatOpenAI

# Configure your LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create an agent with skills
agent = Agent(
    llm=llm,
    skills_dirs=["./skills"],
    agent_config_path="./agent.md"
)

# Run the agent
response = agent.run("Help me create a budget for retirement savings")
print(response)
```

## Skill Format

Skills follow the [Agent Skills specification](https://agentskills.io/):

```markdown
---
name: financial-advisor
description: "Provides budgeting and expense optimization advice."
---

## Instructions

Use this skill to analyze income and expenses and provide actionable suggestions...
```

### Skill Directory Structure

```
skills/
├── financial-advisor/
│   ├── SKILL.md          # Required: skill metadata + instructions
│   ├── scripts/          # Optional: executable code
│   │   └── calculate.py
│   ├── references/       # Optional: detailed documentation
│   │   └── methodology.md
│   └── assets/           # Optional: templates, data files
│       └── budget_template.xlsx
```

## Agent Configuration

Create an `agent.md` file to define your agent's personality and capabilities:

```markdown
# Agent Configuration

## Primary Specialization
You are a Financial Advisor AI assistant specialized in budgeting and investment advice.

## Core Capabilities
1. Financial planning and budget optimization
2. Investment portfolio analysis
3. Retirement planning strategies

## Behavior
- Be concise and actionable
- Ask clarifying questions when needed
- Provide step-by-step guidance
```

## Progressive Disclosure

The library implements progressive disclosure to minimize token usage:

1. **Startup** - Load skill metadata (name + description) for all skills
2. **Selection** - Match user query to relevant skills based on descriptions
3. **Activation** - Load full skill instructions when needed
4. **Resources** - Load scripts, references, and assets on-demand

## LLM Configuration Best Practices

Pass pre-configured LLM instances for maximum flexibility:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from agents_with_intent import Agent

# OpenAI
openai_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
agent1 = Agent(llm=openai_llm, skills_dirs=["./skills"])

# Anthropic
anthropic_llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
agent2 = Agent(llm=anthropic_llm, skills_dirs=["./skills"])

# Ollama (self-hosted)
ollama_llm = ChatOpenAI(
    model="llama3.1:70b",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
agent3 = Agent(llm=ollama_llm, skills_dirs=["./skills"])
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=agents_with_intent
```

## Architecture

The library uses LangGraph to create a state machine with these nodes:

- **skill_discovery** - Scan directories and load skill metadata
- **skill_selection** - Match user input to relevant skills
- **skill_activation** - Load full skill content into context
- **llm_generation** - Generate responses using configured LLM
- **tool_execution** - Execute tools bound to the LLM
- **output** - Format and return final response

State is persisted using LangGraph's MemorySaver for multi-turn conversations.

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Related Projects

- [DeepAgent-Skills](https://github.com/Tomer-Gal/DeepAgent-Skills) - Example implementation using agents-with-intent
- [Agent Skills Specification](https://agentskills.io/) - Official skill format spec
- [LangGraph](https://github.com/langchain-ai/langgraph) - State machine framework