"""Agents-With-Intent: LangGraph-based AI agents with progressive skill discovery.

This library provides a framework for building AI agents that follow the
official Agent Skills specification (https://agentskills.io/).
"""

from agents_with_intent.agent import Agent
from agents_with_intent.skills.discovery import discover_skills
from agents_with_intent.skills.parser import parse_skill_metadata as parse_skill
from agents_with_intent.skills.loader import SkillLoader

__version__ = "0.1.0"
__all__ = ["Agent", "discover_skills", "parse_skill", "SkillLoader"]
