"""Agents-With-Intent: LangGraph-based AI agents with progressive skill discovery.

This library provides a framework for building AI agents that follow the
official Agent Skills specification (https://agentskills.io/).

Two architectures are available:
1. Flat Accumulator (create_agent_graph): Simple loop with Select -> Generate -> Tools
2. Hierarchical Supervisor (create_supervisor_graph): Supervisor -> Worker model
"""

from agents_with_intent.agent import Agent
from agents_with_intent.skills.discovery import discover_skills
from agents_with_intent.skills.parser import parse_skill_metadata as parse_skill
from agents_with_intent.skills.loader import SkillLoader
from agents_with_intent.graph.builder import create_agent_graph, create_supervisor_graph
from agents_with_intent.graph.supervisor import supervisor_node, route_supervisor
from agents_with_intent.standard_tools import get_standard_tools, STANDARD_TOOLS

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "discover_skills",
    "parse_skill",
    "SkillLoader",
    # Graph builders
    "create_agent_graph",
    "create_supervisor_graph",
    # Supervisor
    "supervisor_node",
    "route_supervisor",
    # Standard tools
    "get_standard_tools",
    "STANDARD_TOOLS",
]

