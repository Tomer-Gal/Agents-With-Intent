from __future__ import annotations

import time
from pathlib import Path

import pytest

from agents_with_intent.skills.registry import SkillRegistry


def _write_skill(skill_dir: Path, *, frontmatter: str, body: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(frontmatter + "\n\n" + body + "\n", encoding="utf-8")


def test_skill_discovery(tmp_path: Path) -> None:
    skills_root = tmp_path / "tmp_skills"

    _write_skill(
        skills_root / "skill-a",
        frontmatter="---\nname: skill-a\ndescription: Skill A does A\n---",
        body="## Instructions\n\nDo A.",
    )

    _write_skill(
        skills_root / "skill-b",
        frontmatter="---\nname: skill-b\n---",
        body="## Instructions\n\nDo B.",
    )

    # Should be ignored (no SKILL.md)
    (skills_root / "empty_folder").mkdir(parents=True, exist_ok=True)

    registry = SkillRegistry(skills_dir=skills_root)
    registry.discover()

    assert registry.get_skill("skill-a").name == "skill-a"
    assert registry.get_skill("skill-b").description == "No description"

    # exactly 2 skills
    assert len(registry.list_skills().splitlines()) == 2

    assert registry.get_skill("skill-a").instructions == "## Instructions\n\nDo A."


def test_discovery_performance_under_100(tmp_path: Path) -> None:
    skills_root = tmp_path / "tmp_skills"
    skills_root.mkdir(parents=True, exist_ok=True)

    for i in range(80):
        _write_skill(
            skills_root / f"skill_{i:02d}",
            frontmatter=f"---\nname: skill_{i:02d}\ndescription: d\n---",
            body="Long body\n" * 5,
        )

    registry = SkillRegistry(skills_dir=skills_root)

    t0 = time.perf_counter()
    registry.discover()
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5
