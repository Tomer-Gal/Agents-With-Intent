"""Skill registry implementing Agent Skills spec with progressive disclosure.

This module provides a registry that:
- Discovers SKILL.md files under a skills directory (recursively)
- Loads only frontmatter metadata for listing skills (Level 1)
- Loads full markdown instructions on-demand (Level 2)
- Provides sandboxed file access inside a skill directory

Parsing uses `python-frontmatter` (import name: `frontmatter`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import frontmatter


class SecurityError(ValueError):
    """Raised when a path attempts to escape a skill directory."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    path: Path
    version: Optional[str] = None


class SkillRegistry:
    def __init__(self, skills_dir: str | Path = "./skills"):
        self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, Skill] = {}

    def discover(self) -> None:
        """Scans for SKILL.md files and populates internal skill map."""
        self._skills = {}

        for skill_file in self.skills_dir.glob("**/SKILL.md"):
            post = frontmatter.load(skill_file)

            # Ensure 'name' is in frontmatter, else use folder name
            name = post.metadata.get("name", skill_file.parent.name)
            description = post.metadata.get("description", "No description")
            version = post.metadata.get("version")

            instructions = (post.content or "").strip()

            self._skills[str(name)] = Skill(
                name=str(name),
                description=str(description),
                instructions=instructions,
                path=skill_file.parent,
                version=str(version) if version is not None else None,
            )

    def get_skill(self, name: str) -> Skill:
        return self._skills[name]

    def list_skills(self) -> str:
        """Returns formatted list of Name: Description for the system prompt."""
        return "\n".join([f"- {s.name}: {s.description}" for s in self._skills.values()])

    def read_skill_resource(self, skill_name: str, file_path: str | Path) -> str:
        """Read a file within a skill directory with strict sandboxing."""
        skill = self.get_skill(skill_name)
        skill_root = skill.path.resolve()

        candidate = (skill_root / Path(file_path)).resolve()
        try:
            candidate.relative_to(skill_root)
        except Exception as e:
            raise SecurityError(
                f"Access denied: '{file_path}' is outside skill '{skill_name}'"
            ) from e

        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(str(candidate))

        return candidate.read_text(encoding="utf-8")
