"""Standard library tools for agent workers.

These tools provide common functionality that can be requested by any skill
via the 'tools' field in their SKILL.md metadata.

Available standard tools:
- file_read: Read contents of a file
- file_write: Write contents to a file
- file_append: Append contents to a file
- file_list: List files in a directory
- web_search: Search the web (placeholder)
- calculate: Perform basic calculations
"""
from typing import List, Optional
from pathlib import Path
import os
import math
import subprocess

from langchain_core.tools import tool


def _file_root() -> Path:
    """Return the root directory that file tools are allowed to access.

    Default is current working directory. Override with
    AGENTS_WITH_INTENT_FILE_ROOT.
    """
    root = os.environ.get("AGENTS_WITH_INTENT_FILE_ROOT")
    base = Path(root).expanduser() if root else Path.cwd()
    return base.resolve()


def _resolve_path_in_root(user_path: str) -> Path:
    """Resolve a user-provided path and ensure it stays within _file_root()."""
    root = _file_root()
    candidate = Path(user_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate

    candidate = candidate.resolve()

    try:
        candidate.relative_to(root)
    except Exception as e:
        raise PermissionError(
            f"Access denied: '{user_path}' is outside allowed root '{root}'"
        ) from e

    return candidate


# =============================================================================
# File Operations
# =============================================================================

@tool
def file_read(file_path: str) -> str:
    """Read the contents of a file.
    
    Args:
        file_path: Path to the file to read (relative or absolute)
        
    Returns:
        File contents as string, or error message
    """
    try:
        path = _resolve_path_in_root(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"
        if not path.is_file():
            return f"Error: Not a file: {file_path}"
        
        # Read with size limit for safety
        max_size = 1024 * 1024  # 1MB limit
        if path.stat().st_size > max_size:
            return f"Error: File too large (max 1MB): {file_path}"
        
        return path.read_text(encoding='utf-8')
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def file_write(file_path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success message or error
    """
    try:
        path = _resolve_path_in_root(file_path)
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        path.write_text(content, encoding='utf-8')
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def file_append(file_path: str, content: str) -> str:
    """Append content to a file. Creates the file if it doesn't exist.
    
    Args:
        file_path: Path to the file to append to
        content: Content to append to the file
        
    Returns:
        Success message or error
    """
    try:
        path = _resolve_path_in_root(file_path)
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully appended {len(content)} characters to {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error appending to file: {e}"


@tool
def file_list(directory_path: str, pattern: Optional[str] = None) -> str:
    """List files in a directory.
    
    Args:
        directory_path: Path to the directory to list
        pattern: Optional glob pattern to filter files (e.g., "*.txt")
        
    Returns:
        List of files as newline-separated string, or error message
    """
    try:
        path = _resolve_path_in_root(directory_path)
        if not path.exists():
            return f"Error: Directory not found: {directory_path}"
        if not path.is_dir():
            return f"Error: Not a directory: {directory_path}"
        
        if pattern:
            files = list(path.glob(pattern))
        else:
            files = list(path.iterdir())
        
        # Sort and format
        files.sort()
        result = []
        for f in files[:100]:  # Limit to 100 entries
            prefix = "[DIR] " if f.is_dir() else "[FILE]"
            result.append(f"{prefix} {f.name}")
        
        if len(files) > 100:
            result.append(f"... and {len(files) - 100} more items")
        
        return "\n".join(result) if result else "Directory is empty"
    except PermissionError:
        return f"Error: Permission denied: {directory_path}"
    except Exception as e:
        return f"Error listing directory: {e}"


# =============================================================================
# Calculation Tools
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """Perform a mathematical calculation.
    
    Supports basic arithmetic (+, -, *, /, **), parentheses, and math functions
    like sqrt, sin, cos, tan, log, exp, abs, round, floor, ceil.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)")
        
    Returns:
        Result of the calculation or error message
    """
    # Safe math functions to allow
    safe_dict = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        # Math module functions
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'floor': math.floor,
        'ceil': math.ceil,
        'factorial': math.factorial,
        'pi': math.pi,
        'e': math.e,
    }
    
    try:
        # Basic sanitization - only allow safe characters
        allowed_chars = set('0123456789+-*/().,%^ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        # Evaluate with restricted globals
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Math domain error - {e}"
    except SyntaxError:
        return "Error: Invalid expression syntax"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Search Tools (Placeholder)
# =============================================================================

@tool
def web_search(query: str) -> str:
    """Search the web for information.
    
    Note: This is a placeholder. In production, integrate with a search API.
    
    Args:
        query: Search query
        
    Returns:
        Search results or placeholder message
    """
    return f"Web search not implemented. Query was: {query}\n\nTo enable web search, integrate with a search API like Serper, Tavily, or similar."


# =============================================================================
# Command Execution (Allowlisted)
# =============================================================================


@tool
def run_command(
    program: str,
    arguments: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    timeout_s: int = 60,
) -> str:
    """Run a command without a shell using an allowlist.

    This is intentionally conservative. It is meant to support skills that
    need to invoke language runtimes like Python/Node to generate artifacts.

    Allowed programs: python, python3, node

    Args:
        program: Executable name or path (basename must be allowlisted)
        arguments: List of command-line arguments (no shell parsing)
        cwd: Working directory (must be within AGENTS_WITH_INTENT_FILE_ROOT or CWD)
        timeout_s: Timeout in seconds

    Returns:
        Combined stdout/stderr and exit code summary.
    """
    allowed = {"python", "python3", "node"}
    prog_name = Path(program).name
    if prog_name not in allowed:
        return f"Error: program '{prog_name}' not allowed (allowed: {sorted(allowed)})"

    try:
        root = _file_root()
        run_cwd = _resolve_path_in_root(cwd) if cwd else root

        cmd: List[str] = [program]
        if arguments:
            cmd.extend([str(a) for a in arguments])

        result = subprocess.run(
            cmd,
            cwd=str(run_cwd),
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
            check=False,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout
        if stderr:
            combined += ("\n" if combined else "") + stderr

        # Avoid dumping huge outputs into context.
        max_chars = 20_000
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n...<output truncated>..."

        return f"exit_code={result.returncode}\n{combined}".strip()
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_s}s"
    except PermissionError as e:
        return f"Error: {e}"
    except FileNotFoundError:
        return f"Error: program not found: {program}"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Tool Registry
# =============================================================================

# Registry of all standard tools
STANDARD_TOOLS = {
    "file_read": file_read,
    "file_write": file_write,
    "file_append": file_append,
    "file_list": file_list,
    "run_command": run_command,
    "calculate": calculate,
    "web_search": web_search,
}

# Tool categories for documentation
TOOL_CATEGORIES = {
    "file": ["file_read", "file_write", "file_append", "file_list"],
    "exec": ["run_command"],
    "math": ["calculate"],
    "search": ["web_search"],
}


def get_standard_tools() -> List:
    """Get all available standard tools.
    
    Returns:
        List of all standard tool functions
    """
    return list(STANDARD_TOOLS.values())


def get_tool_by_name(name: str):
    """Get a specific standard tool by name.
    
    Args:
        name: Name of the tool
        
    Returns:
        Tool function or None if not found
    """
    return STANDARD_TOOLS.get(name)


def get_tools_by_names(names: List[str]) -> List:
    """Get multiple standard tools by name.
    
    Args:
        names: List of tool names
        
    Returns:
        List of tool functions (skips unknown names)
    """
    tools = []
    for name in names:
        tool_fn = STANDARD_TOOLS.get(name)
        if tool_fn:
            tools.append(tool_fn)
    return tools


def get_tools_by_category(category: str) -> List:
    """Get all standard tools in a category.
    
    Args:
        category: Category name ("file", "math", "search")
        
    Returns:
        List of tool functions in that category
    """
    tool_names = TOOL_CATEGORIES.get(category, [])
    return get_tools_by_names(tool_names)


def list_available_tools() -> str:
    """Get a formatted list of all available standard tools.
    
    Returns:
        Formatted string describing all tools
    """
    lines = ["Available Standard Tools:", ""]
    
    for category, tool_names in TOOL_CATEGORIES.items():
        lines.append(f"## {category.title()}")
        for name in tool_names:
            tool_fn = STANDARD_TOOLS.get(name)
            if tool_fn:
                doc = tool_fn.__doc__ or "No description"
                first_line = doc.strip().split('\n')[0]
                lines.append(f"- {name}: {first_line}")
        lines.append("")
    
    return "\n".join(lines)
