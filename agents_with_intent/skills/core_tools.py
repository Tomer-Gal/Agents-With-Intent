"""Core tools for progressive disclosure.

These tools provide a stable tool schema for the LLM:
- load_skill(skill_name): page-in a skill's full instructions
- read_skill_resource(skill_name, file_path): read a file inside a skill folder

The LangGraph tool execution node is responsible for enforcing sandboxing and
updating state (loaded_skills, etc.).
"""

from langchain_core.tools import tool


@tool
def load_skill(skill_name: str) -> str:
    """Load a skill's full instructions into the agent context.

    Args:
        skill_name: Name of the skill to load.

    Returns:
        A confirmation message.
    """

    return f"load_skill requested for '{skill_name}'"


@tool
def read_skill_resource(skill_name: str, file_path: str) -> str:
    """Read a resource file from within a skill directory.

    Args:
        skill_name: Name of the skill.
        file_path: Relative path within the skill directory.

    Returns:
        File contents as text.
    """

    return f"read_skill_resource requested for '{skill_name}': {file_path}"