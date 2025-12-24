"""Skill loader module for managing skill lifecycle and progressive disclosure."""
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agents_with_intent.skills.parser import (
    parse_skill_full,
    load_reference_file
)
from agents_with_intent.skills.discovery import (
    list_skill_scripts,
    list_skill_references,
    list_skill_assets
)


class SkillLoader:
    """Manages skill loading and execution with progressive disclosure.
    
    This class handles:
    1. Loading full skill instructions when activated
    2. Executing scripts from scripts/ directory
    3. Loading reference documents on-demand
    4. Accessing asset files
    """
    
    def __init__(self, skill_metadata: Dict[str, any]):
        """Initialize skill loader.
        
        Args:
            skill_metadata: Skill metadata from discover_skills()
        """
        self.name = skill_metadata['name']
        self.description = skill_metadata['description']
        self.skill_file = skill_metadata['path']
        self.skill_dir = skill_metadata['skill_dir']
        self.has_scripts = skill_metadata['has_scripts']
        self.has_references = skill_metadata['has_references']
        self.has_assets = skill_metadata['has_assets']
        self.license = skill_metadata.get('license')
        self.compatibility = skill_metadata.get('compatibility')
        self.tools = skill_metadata.get('tools', [])
        self.metadata = skill_metadata.get('metadata', {})
        
        # Full content loaded on-demand
        self._instructions: Optional[str] = None
        self._references_cache: Dict[str, str] = {}
    
    def load_instructions(self) -> str:
        """Load full skill instructions from SKILL.md.
        
        This is called when the skill is activated.
        
        Returns:
            Full markdown content after YAML frontmatter
        """
        if self._instructions is None:
            _, self._instructions = parse_skill_full(self.skill_file)
        return self._instructions
    
    def get_scripts(self) -> List[Path]:
        """Get list of available scripts in scripts/ directory.
        
        Returns:
            List of paths to script files
        """
        if not self.has_scripts:
            return []
        return list_skill_scripts(self.skill_dir)
    
    def execute_script(
        self,
        script_name: str,
        args: Optional[List[str]] = None,
        timeout: int = 30
    ) -> Tuple[int, str, str]:
        """Execute a script from the scripts/ directory.
        
        Args:
            script_name: Name of script file (e.g., "calculate.py")
            args: Optional list of command-line arguments
            timeout: Execution timeout in seconds (default: 30)
            
        Returns:
            Tuple of (return_code, stdout, stderr)
            
        Raises:
            FileNotFoundError: If script doesn't exist
            subprocess.TimeoutExpired: If execution exceeds timeout
        """
        if not self.has_scripts:
            raise FileNotFoundError(f"Skill '{self.name}' has no scripts/ directory")
        
        script_path = self.skill_dir / "scripts" / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Build command
        cmd = [str(script_path)]
        if args:
            cmd.extend(args)
        
        # Execute with timeout
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.skill_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # Don't raise on non-zero exit
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            raise subprocess.TimeoutExpired(
                cmd=cmd,
                timeout=timeout,
                output=e.stdout,
                stderr=e.stderr
            ) from e
    
    def get_references(self) -> List[Path]:
        """Get list of available reference documents.
        
        Returns:
            List of paths to reference files
        """
        if not self.has_references:
            return []
        return list_skill_references(self.skill_dir)
    
    def load_reference(self, reference_name: str) -> str:
        """Load a reference document by name.
        
        Args:
            reference_name: Name of reference file relative to references/
                           (e.g., "methodology.md" or "guides/advanced.md")
        
        Returns:
            Content of the reference file
            
        Raises:
            FileNotFoundError: If reference doesn't exist
        """
        if not self.has_references:
            raise FileNotFoundError(f"Skill '{self.name}' has no references/ directory")
        
        # Check cache first
        if reference_name in self._references_cache:
            return self._references_cache[reference_name]
        
        reference_path = self.skill_dir / "references" / reference_name
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference not found: {reference_path}")
        
        # Load and cache
        content = load_reference_file(reference_path)
        self._references_cache[reference_name] = content
        return content
    
    def get_assets(self) -> List[Path]:
        """Get list of available asset files.
        
        Returns:
            List of paths to asset files
        """
        if not self.has_assets:
            return []
        return list_skill_assets(self.skill_dir)
    
    def get_asset_path(self, asset_name: str) -> Path:
        """Get path to an asset file.
        
        Args:
            asset_name: Name of asset file relative to assets/
                       (e.g., "template.xlsx" or "images/diagram.png")
        
        Returns:
            Path to the asset file
            
        Raises:
            FileNotFoundError: If asset doesn't exist
        """
        if not self.has_assets:
            raise FileNotFoundError(f"Skill '{self.name}' has no assets/ directory")
        
        asset_path = self.skill_dir / "assets" / asset_name
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset not found: {asset_path}")
        
        return asset_path
    
    def to_prompt_context(self, include_instructions: bool = True) -> str:
        """Convert skill to prompt context string.
        
        Args:
            include_instructions: Whether to include full instructions
                                 (False for metadata-only progressive disclosure)
        
        Returns:
            Formatted string for inclusion in system prompt
        """
        context = f"<skill>\n"
        context += f"  <name>{self.name}</name>\n"
        context += f"  <description>{self.description}</description>\n"
        
        if self.tools:
            context += f"  <tools>{' '.join(self.tools)}</tools>\n"
        
        if include_instructions:
            instructions = self.load_instructions()
            context += f"\n<instructions>\n{instructions}\n</instructions>\n"
        
        context += f"</skill>"
        return context
