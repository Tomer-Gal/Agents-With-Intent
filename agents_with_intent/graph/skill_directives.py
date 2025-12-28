"""Skill instruction parser - automatically execute MANDATORY directives.

This module provides a preprocessing node for the LangGraph that parses loaded
skill instructions and automatically executes mandatory directives like:
- MANDATORY - READ ENTIRE FILE: Read [file.md]
- CRITICAL: Execute script.py

This solves the problem where LLMs read skill documentation but don't interpret
markdown formatting as executable commands.
"""

import re
from typing import Dict, List
from pathlib import Path
import logging

from agents_with_intent.graph.state import AgentState
from agents_with_intent.skills.loader import SkillLoader

logger = logging.getLogger(__name__)


def parse_mandatory_directives(skill_content: str, skill_path: Path) -> List[Dict]:
    """Parse skill content for MANDATORY/CRITICAL directives and extract file read instructions.
    
    Args:
        skill_content: Full SKILL.md content
        skill_path: Path to the skill directory (for resolving relative paths)
        
    Returns:
        List of directive dicts with 'action', 'file_path', 'reason'
    """
    directives = []
    
    # Pattern: **MANDATORY** or **CRITICAL** followed by "Read" and a markdown link
    # Examples:
    # 1. **MANDATORY - READ ENTIRE FILE**: Read [`html2pptx.md`](html2pptx.md)
    # 2. **CRITICAL**: Read [documentation.md]
    # 3. **REQUIRED**: Read file.md completely
    
    patterns = [
        # Captures: **MANDATORY - READ ENTIRE FILE**: Read [`html2pptx.md`](html2pptx.md)
        r'\*\*(MANDATORY|CRITICAL|REQUIRED)[^*]*\*\*[^:]*:\s*Read\s*\[`?([^\]`]+)`?\]\(([^\)]+)\)',
        # Captures: **MANDATORY**: Read [file.md]
        r'\*\*(MANDATORY|CRITICAL|REQUIRED)\*\*[^:]*:\s*Read\s*\[([^\]]+)\]',
        # Captures: **MANDATORY** Read file.md
        r'\*\*(MANDATORY|CRITICAL|REQUIRED)\*\*[^R]*Read\s+([a-zA-Z0-9_\-\.\/]+\.md)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, skill_content, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            keyword = match.group(1)
            # Extract filename from different capture groups depending on pattern
            if len(match.groups()) >= 3:
                filename = match.group(3) if match.group(3) else match.group(2)
            else:
                filename = match.group(2)
            
            # Resolve relative path from skill directory
            if not filename.startswith('/'):
                file_path = skill_path / filename
            else:
                file_path = Path(filename)
            
            directives.append({
                'action': 'file_read',
                'file_path': str(file_path),
                'reason': f'{keyword} directive in skill documentation',
                'read_all': 'READ ENTIRE FILE' in match.group(0).upper()
            })
            
            logger.info(f"Found {keyword} directive: Read {file_path}")
    
    return directives


def skill_directive_execution_node(state: AgentState) -> Dict:
    """Execute mandatory directives from loaded skills before LLM generation.
    
    This node runs after skill loading and automatically executes MANDATORY/CRITICAL
    directives found in skill documentation, such as reading referenced files.
    
    Args:
        state: Current agent state
        
    Returns:
        State update with executed directives stored in messages
    """
    loaded_skills = state.get("loaded_skills", [])
    skill_loaders = state.get("skill_loaders", {})
    
    if not loaded_skills or not skill_loaders:
        return {}
    
    messages = []
    
    for skill_name in loaded_skills:
        if skill_name not in skill_loaders:
            continue
        
        # Get skill metadata and create loader
        meta = skill_loaders[skill_name]
        loader = SkillLoader(meta)
        
        # Get full skill instructions
        instructions = loader.load_instructions()
        skill_path = Path(meta['path'])
        
        # Parse for mandatory directives
        directives = parse_mandatory_directives(instructions, skill_path)
        
        if not directives:
            continue
        
        logger.info(f"Executing {len(directives)} mandatory directives from skill '{skill_name}'")
        
        # Execute each directive
        for directive in directives:
            if directive['action'] == 'file_read':
                file_path = Path(directive['file_path'])
                
                if not file_path.exists():
                    logger.warning(f"MANDATORY file not found: {file_path}")
                    continue
                
                try:
                    # Read the entire file as instructed
                    content = file_path.read_text(encoding='utf-8')
                    
                    # Add a system message documenting that this was read
                    from langchain_core.messages import SystemMessage
                    msg = SystemMessage(content=f"""
[AUTO-EXECUTED MANDATORY DIRECTIVE from skill '{skill_name}']
Read file: {file_path}
Reason: {directive['reason']}

File content ({len(content)} chars):
---
{content}
---
""")
                    messages.append(msg)
                    
                    logger.info(f"✓ Executed: Read {file_path} ({len(content)} chars)")
                    
                except Exception as e:
                    logger.error(f"Failed to read mandatory file {file_path}: {e}")
    
    if messages:
        return {"messages": messages}
    
    return {}
