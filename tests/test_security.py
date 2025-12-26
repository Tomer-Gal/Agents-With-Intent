from __future__ import annotations

from pathlib import Path

import pytest

from agents_with_intent.skills.registry import SecurityError, SkillRegistry


def _write_skill(skill_dir: Path) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: test\n---\n\n## Instructions\n\nHi\n",
        encoding="utf-8",
    )


def test_read_skill_resource_allows_only_within_skill(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "skill-a"

    _write_skill(skill_dir)
    (skill_dir / "template.txt").write_text("OK", encoding="utf-8")

    secret = tmp_path / "secret.txt"
    secret.write_text("NO", encoding="utf-8")

    reg = SkillRegistry(skills_dir=skills_root)
    reg.discover()

    assert reg.read_skill_resource("skill-a", "template.txt") == "OK"

    with pytest.raises((SecurityError, ValueError)):
        reg.read_skill_resource("skill-a", "../secret.txt")

    with pytest.raises(SecurityError):
        reg.read_skill_resource("skill-a", "/etc/passwd")
