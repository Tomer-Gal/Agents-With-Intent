"""Convert skill scripts into LangChain tools for proper tool-calling integration."""
from typing import List, Callable
from pathlib import Path
import subprocess

from langchain_core.tools import tool

from agents_with_intent.skills.loader import SkillLoader


def create_script_tool(
    skill_loader: SkillLoader,
    script_path: Path
) -> Callable:
    """Create a LangChain tool from a skill script.
    
    Args:
        skill_loader: Skill loader instance
        script_path: Path to the script file
        
    Returns:
        LangChain @tool decorated function
    """
    script_name = script_path.name
    skill_name = skill_loader.name
    
    # Create tool function with proper docstring
    def execute_skill_script(arguments: str = "") -> str:
        """Execute script from skill.
        
        Args:
            arguments: Arguments to pass to the script (as single string or space-separated)
            
        Returns:
            Script output (stdout) or error message
        """
        try:
            # For scripts that expect a single argument (like execute_python.py),
            # pass the entire arguments string as one arg
            # For other scripts, split by spaces
            if arguments.strip():
                # Check if script name suggests it needs single argument
                if 'execute' in script_name or 'python' in script_name:
                    # Pass as single argument (for code/large content)
                    args = [arguments]
                else:
                    # Split by spaces (traditional CLI args)
                    args = arguments.split()
            else:
                args = None
            
            # Execute script
            returncode, stdout, stderr = skill_loader.execute_script(
                script_name=script_name,
                args=args,
                timeout=60
            )
            
            if returncode == 0:
                return stdout if stdout else "Script executed successfully (no output)"
            else:
                error_msg = f"Script failed with exit code {returncode}"
                if stderr:
                    error_msg += f"\nError: {stderr}"
                return error_msg
                
        except subprocess.TimeoutExpired:
            return f"Script execution timed out (exceeded 60 seconds)"
        except FileNotFoundError as e:
            return f"Script not found: {e}"
        except Exception as e:
            return f"Execution error: {e}"
    
    # Set proper docstring
    execute_skill_script.__doc__ = f"Execute {script_name} from {skill_name} skill."
    
    # Set unique tool name (replace dots/dashes with underscores)
    tool_name = f"{skill_name}_{script_name}".replace('.', '_').replace('-', '_')
    execute_skill_script.__name__ = tool_name
    
    # Apply @tool decorator and return
    return tool(execute_skill_script)


def create_script_tools(skill_loader: SkillLoader) -> List[Callable]:
    """Create LangChain tools for all scripts in a skill.
    
    Args:
        skill_loader: Skill loader instance with scripts
        
    Returns:
        List of LangChain tool functions
    """
    tools = []
    
    if not skill_loader.has_scripts:
        return tools
    
    for script_path in skill_loader.get_scripts():
        tool_func = create_script_tool(skill_loader, script_path)
        tools.append(tool_func)
    
    return tools


def create_tools_from_active_skills(
    active_skills: List[str],
    skill_loaders: dict
) -> List[Callable]:
    """Create tools from all active skills.
    
    Args:
        active_skills: List of active skill names
        skill_loaders: Dict mapping skill names to skill metadata
        
    Returns:
        List of all tools from active skills
    """
    all_tools = []
    
    for skill_name in active_skills:
        if skill_name in skill_loaders:
            # Create loader from metadata
            loader = SkillLoader(skill_loaders[skill_name])
            
            # Generate tools from scripts
            tools = create_script_tools(loader)
            all_tools.extend(tools)
    
    return all_tools
