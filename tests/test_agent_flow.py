from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from agents_with_intent.graph.nodes import build_system_prompt, tool_execution_node
from agents_with_intent.skills.discovery import discover_skills


def _write_skill(skill_dir: Path, *, name: str, description: str, body: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n" + f"name: {name}\n" + f"description: {description!r}\n" + "---\n\n" + body + "\n",
        encoding="utf-8",
    )


def test_progressive_disclosure_load_skill_updates_state_and_prompt(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"

    _write_skill(
        skills_root / "skill-a",
        name="skill-a",
        description="Alpha skill",
        body="## Instructions\n\nDo A in detail.",
    )

    skills_metadata = discover_skills([str(skills_root)])

    state = {
        "messages": [],
        "skills_metadata": skills_metadata,
        "active_skills": [],
        "loaded_skills": [],
        "skill_loaders": {},
        "agent_config": None,
        "next_action": None,
    }

    prompt_level1 = build_system_prompt(state)
    assert "Alpha skill" in prompt_level1
    assert "Do A in detail." not in prompt_level1

    token_count_level1 = len(prompt_level1.split())

    # Simulate the agent calling load_skill("skill_a")
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[{"name": "load_skill", "args": {"skill_name": "skill-a"}, "id": "1"}],
        )
    ]

    update = tool_execution_node(state)

    # Apply update the same way LangGraph would
    state["loaded_skills"] = update["loaded_skills"]
    state["active_skills"] = update["active_skills"]
    state["skill_loaders"] = update["skill_loaders"]

    assert "skill-a" in state["loaded_skills"]

    prompt_level2 = build_system_prompt(state)
    assert "Do A in detail." in prompt_level2

    token_count_level2 = len(prompt_level2.split())
    assert token_count_level2 > token_count_level1
