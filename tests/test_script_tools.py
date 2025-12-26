#!/usr/bin/env python3
"""Test script tool generation from skills."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agents_with_intent.skills.discovery import discover_skills
from agents_with_intent.skills.loader import SkillLoader
from agents_with_intent.skills.tools import create_script_tools


def test_tool_generation():
    """Test that tools are created from skill scripts."""
    # Discover PDF skill
    skills_dir = Path(__file__).parent.parent.parent / "DeepAgent-Skills" / ".deepagents" / "skills"
    
    if not skills_dir.exists():
        pytest.skip(f"Skills directory not found: {skills_dir}")
    
    skills_metadata = discover_skills([str(skills_dir)])
    
    # Find PDF skill
    pdf_skill = None
    for skill_meta in skills_metadata:
        if skill_meta.get('name') == 'pdf':
            pdf_skill = skill_meta
            break
    
    if not pdf_skill:
        pytest.skip("PDF skill not found in discovered skills")
    
    # Create loader and tools
    loader = SkillLoader(pdf_skill)
    tools = create_script_tools(loader)
    
    print(f"✓ Created {len(tools)} tools from PDF skill")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")

    assert tools, "Expected at least one tool to be created from PDF skill scripts"
    
    # Test tool invocation
    first_tool = tools[0]
    print(f"\n✓ Testing {first_tool.name}...")
    assert hasattr(first_tool, "invoke") or callable(getattr(first_tool, "func", None))
    assert getattr(first_tool, "args", None) is not None


if __name__ == "__main__":
    raise SystemExit("Run via pytest")
