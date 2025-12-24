"""Skill parser module for extracting YAML frontmatter and content from SKILL.md files."""
import re
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional


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
    content = skill_file.read_text(encoding='utf-8')
    
    # Extract YAML frontmatter
    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError(f"No YAML frontmatter found in {skill_file}")
    
    frontmatter_str = frontmatter_match.group(1)
    
    try:
        frontmatter = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {skill_file}: {e}") from e
    
    if not isinstance(frontmatter, dict):
        raise ValueError(f"YAML frontmatter must be a dictionary in {skill_file}")
    
    # Validate required fields per Agent Skills spec
    if 'name' not in frontmatter:
        raise ValueError(f"Missing required 'name' field in {skill_file}")
    
    if 'description' not in frontmatter:
        raise ValueError(f"Missing required 'description' field in {skill_file}")
    
    name = frontmatter['name']
    description = frontmatter['description']
    
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
    
    # Validate description (spec: 1-1024 chars)
    if not isinstance(description, str) or len(description) < 1 or len(description) > 1024:
        raise ValueError(f"Invalid 'description' in {skill_file}: must be 1-1024 characters")
    
    # Validate optional fields
    result = {
        'name': name,
        'description': description,
    }
    
    if 'license' in frontmatter:
        result['license'] = str(frontmatter['license'])
    
    if 'compatibility' in frontmatter:
        compat = str(frontmatter['compatibility'])
        if len(compat) > 500:
            raise ValueError(f"'compatibility' in {skill_file} exceeds 500 characters")
        result['compatibility'] = compat
    
    if 'tools' in frontmatter:
        tools = frontmatter['tools']
        if isinstance(tools, str):
            result['tools'] = tools.split()
        elif isinstance(tools, list):
            result['tools'] = [str(t) for t in tools]
        else:
            raise ValueError(f"'tools' in {skill_file} must be string or list")
    
    if 'metadata' in frontmatter:
        if not isinstance(frontmatter['metadata'], dict):
            raise ValueError(f"'metadata' in {skill_file} must be a dictionary")
        result['metadata'] = frontmatter['metadata']
    
    # Warn about non-standard fields (for backward compatibility)
    standard_fields = {'name', 'description', 'license', 'compatibility', 'tools', 'metadata'}
    non_standard = set(frontmatter.keys()) - standard_fields
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
    content = skill_file.read_text(encoding='utf-8')
    
    # Extract YAML frontmatter and content
    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError(f"No YAML frontmatter found in {skill_file}")
    
    instructions = frontmatter_match.group(2).strip()
    
    # Parse metadata
    metadata = parse_skill_metadata(skill_file)
    
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
