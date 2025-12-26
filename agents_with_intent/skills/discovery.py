"""Skill discovery module for scanning directories and finding SKILL.md files."""
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional


def _cache_enabled() -> bool:
    value = os.environ.get("AGENTS_WITH_INTENT_SKILLS_CACHE")
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _cache_path(skills_dirs: List[str]) -> Path | None:
    value = os.environ.get("AGENTS_WITH_INTENT_SKILLS_CACHE_PATH")
    if value:
        return Path(value).expanduser()

    if not skills_dirs:
        return None

    # Default near the first skills dir so it's naturally persisted when the
    # repo is mounted into a container.
    try:
        first = Path(skills_dirs[0]).resolve()
    except Exception:
        return None
    return first.parent / ".agents_with_intent_skills_cache.json"


def _load_cache(path: Path, skills_dirs: List[str]) -> Dict[str, Dict[str, any]]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        if data.get("version") != 1:
            return {}
        if data.get("skills_dirs") != skills_dirs:
            return {}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            return {}
        # entries: {"/abs/SKILL.md": {"mtime_ns": int, "size": int, "metadata": {...}}}
        return entries
    except Exception:
        return {}


def _save_cache(path: Path, skills_dirs: List[str], entries: Dict[str, Dict[str, any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {"version": 1, "skills_dirs": skills_dirs, "entries": entries}
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Cache failures must never break skill discovery.
        return


def validate_skill_name(name: str) -> bool:
    """Validate skill name according to Agent Skills specification.
    
    Rules:
    - 1-64 characters
    - Lowercase alphanumeric + hyphens only
    - No leading, trailing, or consecutive hyphens
    
    Args:
        name: Skill name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not name or len(name) < 1 or len(name) > 64:
        return False
    
    # Check for leading/trailing hyphens
    if name.startswith('-') or name.endswith('-'):
        return False
    
    # Check for consecutive hyphens
    if '--' in name:
        return False
    
    # Check for valid characters (lowercase alphanumeric + hyphens)
    if not re.match(r'^[a-z0-9-]+$', name):
        return False
    
    return True


def discover_skills(skills_dirs: Optional[List[str]] = None) -> List[Dict[str, any]]:
    """Discover all skills in the specified directories.
    
    Scans directories for SKILL.md files and extracts metadata.
    Implements progressive disclosure - only loads name and description at startup.
    
    Args:
        skills_dirs: List of directory paths to scan for skills.
            If None, defaults to ["./skills"].
        
    Returns:
        List of skill metadata dictionaries with keys:
        - name: Skill name (from YAML frontmatter)
        - description: Skill description (from YAML frontmatter)
        - path: Path to SKILL.md file
        - skill_dir: Parent directory of the skill
        - has_scripts: Whether scripts/ directory exists
        - has_references: Whether references/ directory exists
        - has_assets: Whether assets/ directory exists
        
    Raises:
        ValueError: If skill format is invalid
    """
    from agents_with_intent.skills.parser import parse_skill_metadata
    
    skills = []
    seen_names: set[str] = set()

    if skills_dirs is None:
        skills_dirs = ["./skills"]
    
    cache_entries: Dict[str, Dict[str, any]] = {}
    cache_file: Path | None = None
    if _cache_enabled():
        cache_file = _cache_path(skills_dirs)
        if cache_file is not None:
            cache_entries = _load_cache(cache_file, skills_dirs)

    updated_cache_entries: Dict[str, Dict[str, any]] = dict(cache_entries) if cache_entries else {}

    for skills_dir_str in skills_dirs:
        skills_dir = Path(skills_dir_str).resolve()
        
        if not skills_dir.exists():
            raise ValueError(f"Skills directory not found: {skills_dir}")
        
        if not skills_dir.is_dir():
            raise ValueError(f"Skills path is not a directory: {skills_dir}")
        
        # Find all SKILL.md files (recursive)
        for skill_file in skills_dir.glob("**/SKILL.md"):
            skill_dir = skill_file.parent
            
            # Parse metadata only (progressive disclosure)
            metadata = None
            cache_key = str(skill_file)
            try:
                stat = skill_file.stat()
            except Exception:
                stat = None

            cached = updated_cache_entries.get(cache_key)
            if cached and stat is not None:
                try:
                    if (
                        int(cached.get("mtime_ns")) == int(stat.st_mtime_ns)
                        and int(cached.get("size")) == int(stat.st_size)
                        and isinstance(cached.get("metadata"), dict)
                    ):
                        metadata = cached["metadata"]
                except Exception:
                    metadata = None

            if metadata is None:
                try:
                    metadata = parse_skill_metadata(skill_file)
                except Exception as e:
                    raise ValueError(f"Error parsing skill at {skill_file}: {e}") from e

                if stat is not None:
                    updated_cache_entries[cache_key] = {
                        "mtime_ns": int(stat.st_mtime_ns),
                        "size": int(stat.st_size),
                        "metadata": metadata,
                    }

            # Ensure unique skill names across all directories
            name = metadata["name"]
            if name in seen_names:
                raise ValueError(
                    f"Duplicate skill name '{name}' discovered at {skill_file}. "
                    "Skill names must be unique."
                )
            seen_names.add(name)
            
            # Check for optional directories
            has_scripts = (skill_dir / "scripts").exists()
            has_references = (skill_dir / "references").exists()
            has_assets = (skill_dir / "assets").exists()
            
            skills.append({
                'name': metadata['name'],
                'description': metadata['description'],
                'license': metadata.get('license'),
                'compatibility': metadata.get('compatibility'),
                'tools': metadata.get('tools'),
                'metadata': metadata.get('metadata', {}),
                'path': skill_file,
                'skill_dir': skill_dir,
                'has_scripts': has_scripts,
                'has_references': has_references,
                'has_assets': has_assets,
            })
    
    if cache_file is not None and updated_cache_entries is not None:
        _save_cache(cache_file, skills_dirs, updated_cache_entries)

    return skills


def list_skill_scripts(skill_dir: Path) -> List[Path]:
    """List all executable scripts in a skill's scripts/ directory.
    
    Args:
        skill_dir: Path to skill directory
        
    Returns:
        List of paths to script files
    """
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.exists():
        return []
    
    scripts = []
    for script_file in scripts_dir.iterdir():
        if script_file.is_file() and not script_file.name.startswith('.'):
            scripts.append(script_file)
    
    return scripts


def list_skill_references(skill_dir: Path) -> List[Path]:
    """List all reference documents in a skill's references/ directory.
    
    Args:
        skill_dir: Path to skill directory
        
    Returns:
        List of paths to reference files
    """
    references_dir = skill_dir / "references"
    if not references_dir.exists():
        return []
    
    references = []
    for ref_file in references_dir.rglob("*"):
        if ref_file.is_file() and not ref_file.name.startswith('.'):
            references.append(ref_file)
    
    return references


def list_skill_assets(skill_dir: Path) -> List[Path]:
    """List all asset files in a skill's assets/ directory.
    
    Args:
        skill_dir: Path to skill directory
        
    Returns:
        List of paths to asset files
    """
    assets_dir = skill_dir / "assets"
    if not assets_dir.exists():
        return []
    
    assets = []
    for asset_file in assets_dir.rglob("*"):
        if asset_file.is_file() and not asset_file.name.startswith('.'):
            assets.append(asset_file)
    
    return assets
