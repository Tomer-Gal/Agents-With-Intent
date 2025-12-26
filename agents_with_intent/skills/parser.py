"""Skill parser module for extracting YAML frontmatter and content from SKILL.md files.

Implements the Open Agent Skills Specification (https://agentskills.io/).
Parsing is done via `python-frontmatter` (import name: `frontmatter`).
"""

import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import frontmatter


def parse_skill_metadata(skill_file: Path) -> Dict[str, any]:
    """Parse only the YAML frontmatter from a SKILL.md file.
    
    This implements progressive disclosure - only metadata is loaded initially.
    Full content is loaded later when the skill is activated.
    
    Args:
        skill_file: Path to SKILL.md file
        
    Returns:
        Dictionary with skill metadata:
        - name: Required, 1-64 chars, lowercase alphanumeric + hyphens
        - description: Required, 1-1024 chars
        - license: Optional
        - compatibility: Optional, max 500 chars
        - tools: Optional, space-delimited list
        - metadata: Optional, arbitrary key-value mapping
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    try:
        post = frontmatter.load(skill_file)
    except Exception as e:
        raise ValueError(f"Failed to parse frontmatter in {skill_file}: {e}") from e

    metadata = post.metadata
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(f"No YAML frontmatter found in {skill_file}")
    
    # Progressive disclosure registry may want to operate even when frontmatter
    # is incomplete. Fall back to folder name and a default description.
    name = metadata.get('name', skill_file.parent.name)
    description = metadata.get('description', 'No description')
    
    # Validate name (spec: 1-64 chars, lowercase alphanumeric + hyphens)
    if not isinstance(name, str) or len(name) < 1 or len(name) > 64:
        raise ValueError(f"Invalid 'name' in {skill_file}: must be 1-64 characters")
    
    if not re.match(r'^[a-z0-9-]+$', name):
        raise ValueError(
            f"Invalid 'name' in {skill_file}: must be lowercase alphanumeric + hyphens"
        )
    
    if name.startswith('-') or name.endswith('-') or '--' in name:
        raise ValueError(
            f"Invalid 'name' in {skill_file}: no leading/trailing/consecutive hyphens"
        )
    
    # Validate description (spec: 1-1024 chars). If missing, we default to
    # 'No description' which satisfies the length constraint.
    if not isinstance(description, str) or len(description) < 1 or len(description) > 1024:
        raise ValueError(
            f"Invalid 'description' in {skill_file}: must be 1-1024 characters"
        )
    
    # Validate optional fields
    result = {
        'name': name,
        'description': description,
    }

    if 'version' in metadata:
        result['version'] = str(metadata['version'])

    # Keep backward-compatible support for optional fields.
    if 'license' in metadata:
        result['license'] = str(metadata['license'])

    if 'compatibility' in metadata:
        compat = str(metadata['compatibility'])
        if len(compat) > 500:
            raise ValueError(f"'compatibility' in {skill_file} exceeds 500 characters")
        result['compatibility'] = compat

    if 'tools' in metadata:
        tools = metadata['tools']
        if isinstance(tools, str):
            result['tools'] = tools.split()
        elif isinstance(tools, list):
            result['tools'] = [str(t) for t in tools]
        else:
            raise ValueError(f"'tools' in {skill_file} must be string or list")

    if 'metadata' in metadata:
        if not isinstance(metadata['metadata'], dict):
            raise ValueError(f"'metadata' in {skill_file} must be a dictionary")
        result['metadata'] = metadata['metadata']

    # Warn about non-standard fields (forward-compatible with spec).
    standard_fields = {'name', 'description', 'version', 'license', 'compatibility', 'tools', 'metadata'}
    non_standard = set(metadata.keys()) - standard_fields
    if non_standard:
        import warnings
        warnings.warn(
            f"Non-standard fields in {skill_file}: {non_standard}. "
            f"These are not part of the Agent Skills spec and will be ignored.",
            UserWarning
        )
    
    return result


def parse_skill_full(skill_file: Path) -> Tuple[Dict[str, any], str]:
    """Parse both metadata and full content from a SKILL.md file.
    
    This is called when a skill is activated and full instructions are needed.
    
    Args:
        skill_file: Path to SKILL.md file
        
    Returns:
        Tuple of (metadata_dict, instructions_content)
        - metadata_dict: Same as parse_skill_metadata()
        - instructions_content: Full markdown content after frontmatter
    """
    metadata = parse_skill_metadata(skill_file)
    try:
        post = frontmatter.load(skill_file)
    except Exception as e:
        raise ValueError(f"Failed to parse frontmatter in {skill_file}: {e}") from e

    instructions = (post.content or "").strip()
    return metadata, instructions


def load_reference_file(reference_path: Path) -> str:
    """Load a reference document from a skill's references/ directory.
    
    Args:
        reference_path: Path to reference file
        
    Returns:
        Content of the reference file
    """
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    
    return reference_path.read_text(encoding='utf-8')
