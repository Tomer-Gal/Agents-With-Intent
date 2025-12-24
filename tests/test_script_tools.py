#!/usr/bin/env python3
"""Test script tool generation from skills."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents_with_intent.skills.discovery import discover_skills
from agents_with_intent.skills.loader import SkillLoader
from agents_with_intent.skills.tools import create_script_tools


def test_tool_generation():
    """Test that tools are created from skill scripts."""
    # Discover PDF skill
    skills_dir = Path(__file__).parent.parent.parent / "DeepAgent-Skills" / ".deepagents" / "skills"
    
    if not skills_dir.exists():
        print(f"Skills directory not found: {skills_dir}")
        return False
    
    skills_metadata = discover_skills([str(skills_dir)])
    
    # Find PDF skill
    pdf_skill = None
    for skill_meta in skills_metadata:
        if skill_meta.get('name') == 'pdf':
            pdf_skill = skill_meta
            break
    
    if not pdf_skill:
        print("PDF skill not found")
        return False
    
    # Create loader and tools
    loader = SkillLoader(pdf_skill)
    tools = create_script_tools(loader)
    
    print(f"✓ Created {len(tools)} tools from PDF skill")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")
    
    # Test tool invocation
    if tools:
        first_tool = tools[0]
        print(f"\n✓ Testing {first_tool.name}...")
        print(f"  Tool is callable: {callable(first_tool)}")
        print(f"  Tool schema: {first_tool.args}")
    
    return True


if __name__ == "__main__":
    success = test_tool_generation()
    sys.exit(0 if success else 1)
