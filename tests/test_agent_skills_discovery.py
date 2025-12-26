from __future__ import annotations

import time
from pathlib import Path

import pytest

from agents_with_intent.skills.discovery import discover_skills
from agents_with_intent.skills.parser import parse_skill_metadata, parse_skill_full


def _write_skill(skill_dir: Path, *, name: str, description: str, body: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description!r}\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )


def test_parse_skill_metadata_and_body(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "anything"
    _write_skill(
        skill,
        name="my-skill",
        description="Does X and Y",
        body="## Instructions\n\nDo the thing.",
    )

    metadata = parse_skill_metadata(skill / "SKILL.md")
    assert metadata["name"] == "my-skill"
    assert metadata["description"] == "Does X and Y"

    full_meta, instructions = parse_skill_full(skill / "SKILL.md")
    assert full_meta["name"] == "my-skill"
    assert "Do the thing." in instructions


def test_discover_skills_default_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root / "some-folder",
        name="skill-one",
        description="Keyword: budget",
        body="Use this skill for budgeting.",
    )

    monkeypatch.chdir(tmp_path)

    discovered = discover_skills()
    assert len(discovered) == 1
    assert discovered[0]["name"] == "skill-one"
    assert discovered[0]["description"] == "Keyword: budget"


def test_discover_skills_performance(tmp_path: Path) -> None:
    """Basic perf guardrail without external plugins.

    This isn't a microbenchmark; it ensures discovery doesn't become
    accidentally quadratic or load full bodies.
    """

    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)

    count = 250
    for i in range(count):
        _write_skill(
            skills_root / f"dir-{i:04d}",
            name=f"skill-{i:04d}",
            description="metadata only",
            body="Long body\n" * 50,
        )

    t0 = time.perf_counter()
    discovered = discover_skills([str(skills_root)])
    elapsed = time.perf_counter() - t0

    assert len(discovered) == count
    # Very generous threshold to avoid flakes; meant to catch regressions.
    assert elapsed < 5.0
