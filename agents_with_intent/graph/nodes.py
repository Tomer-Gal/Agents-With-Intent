"""Graph nodes for LangGraph agent state machine."""
from typing import Dict, List, Optional, Callable
from pathlib import Path
import re
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from agents_with_intent.graph.state import AgentState
from agents_with_intent.skills.discovery import discover_skills
from agents_with_intent.skills.loader import SkillLoader
from agents_with_intent.skills.tools import create_tools_from_active_skills
from agents_with_intent.skills.core_tools import load_skill, read_skill_resource
from agents_with_intent.skills.registry import SecurityError


logger = logging.getLogger(__name__)


# Import standard tools if available
try:
    from agents_with_intent.standard_tools import get_standard_tools
except ImportError:
    def get_standard_tools():
        return []


def load_agent_config_node(state: AgentState, agent_config_path: Optional[str]) -> Dict:
    """Load agent configuration from agent.md file.
    
    Args:
        state: Current agent state
        agent_config_path: Path to agent.md file (optional)
        
    Returns:
        State update with agent_config
    """
    if not agent_config_path:
        return {"agent_config": None}
    
    config_path = Path(agent_config_path)
    if not config_path.exists():
        return {"agent_config": None}
    
    agent_config = config_path.read_text(encoding='utf-8')
    return {"agent_config": agent_config}


def discover_skills_node(state: AgentState, skills_dirs: List[str]) -> Dict:
    """Discover and load skill metadata from configured directories.
    
    Implements progressive disclosure - only loads name and description.
    
    Args:
        state: Current agent state
        skills_dirs: List of directories to scan for skills
        
    Returns:
        State update with skills_metadata
    """
    skills_metadata = discover_skills(skills_dirs)
    
    return {
        "skills_metadata": skills_metadata,
        "active_skills": [],
        "loaded_skills": [],
        "skill_loaders": {}
    }


def build_system_prompt(state: AgentState, include_instructions: bool = False) -> str:
    """Build system prompt from agent config and skill metadata.
    
    Args:
        state: Current agent state
        include_instructions: Whether to include full skill instructions
        
    Returns:
        System prompt string
    """
    prompt = ""
    
    # Add agent configuration if available
    if state.get("agent_config"):
        prompt += state["agent_config"]
        prompt += "\n\n"
    
    # Add available skills (Level 1: only name + description)
    skills_metadata = state.get("skills_metadata", [])
    if skills_metadata:
        prompt += "<available_skills>\n"
        for skill_meta in skills_metadata:
            name = skill_meta.get("name")
            description = skill_meta.get("description")
            prompt += f"- {name}: {description}\n"
        prompt += "</available_skills>\n\n"
    
    # Add loaded skill instructions (Level 2)
    # Backward compatibility: older callers may not provide `loaded_skills`.
    # In that case, treat `active_skills` as loaded.
    loaded_skills = state.get("loaded_skills")
    if loaded_skills is None:
        loaded_skills = state.get("active_skills", [])
    skill_loaders = state.get("skill_loaders", {})
    if loaded_skills and skill_loaders:
        prompt += "<activated_skills>\n"
        for skill_name in loaded_skills:
            if skill_name in skill_loaders:
                # skill_loaders stores serializable metadata dicts; create
                # a SkillLoader instance on-demand to access instructions.
                meta = skill_loaders[skill_name]
                loader = SkillLoader(meta)
                instructions = loader.load_instructions()
                prompt += f"<skill name=\"{skill_name}\">\n"
                prompt += f"{instructions}\n"
                prompt += f"</skill>\n\n"
        prompt += "</activated_skills>\n\n"
    
    # Add skill selection guidance
    prompt += """## Skill Selection

When the user's request matches a skill's description:
1. Activate the skill if not already active
2. Follow the skill's instructions carefully
3. Use the skill's scripts, references, and assets as needed

Available skill resources:
- scripts/: Executable code - **You have tools to execute these scripts directly**
- references/: Detailed documentation (load when needed)
- assets/: Templates, data files, images (access when needed)

When a skill has scripts, they are available as tools you can call. Use them to perform actions rather than just providing code examples.
"""
    
    return prompt


def skill_selection_node(state: AgentState) -> Dict:
    """Analyze user input and select relevant skills to activate.
    
    This uses simple keyword matching against skill descriptions.
    For production, consider using semantic similarity.
    
    Args:
        state: Current agent state
        
    Returns:
        State update with active_skills and skill_loaders
    """
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    # Get last user message
    last_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_message = msg.content.lower()
            break
    
    if not last_message:
        return {}
    
    # Get current state
    skills_metadata = state.get("skills_metadata", [])
    active_skills = list(state.get("active_skills", []))
    skill_loaders = dict(state.get("skill_loaders", {}))
    
    def tokenize(text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))

        # Heuristic: treat arithmetic operators as intent hints. This enables
        # prompts like "5+2" to match a skill described with words like "add".
        if "+" in text:
            tokens.update({"plus", "add"})
        if "-" in text:
            tokens.update({"minus", "subtract"})
        if "*" in text or "×" in text:
            tokens.update({"times", "multiply"})
        if "/" in text or "÷" in text:
            tokens.update({"divide", "division"})

        return tokens

    # Simple keyword matching (TODO: use semantic similarity in production)
    for skill_meta in skills_metadata:
        skill_name = skill_meta['name']
        description = skill_meta['description'].lower()
        
        # Skip if already active
        if skill_name in active_skills:
            continue
        
        # Check if any description words appear in user message
        # This is a simple heuristic - production should use embeddings
        desc_words = tokenize(description)
        msg_words = tokenize(last_message)
        
        # If there's overlap, activate the skill
        if desc_words & msg_words:
            active_skills.append(skill_name)
            # Store serializable skill metadata in state; SkillLoader instances
            # are not JSON-serializable for checkpointing, so create loaders
            # on-demand when needed.
            skill_loaders[skill_name] = skill_meta
    
    return {
        "active_skills": active_skills,
        "skill_loaders": skill_loaders
    }


def llm_generation_node(state: AgentState, llm: BaseChatModel) -> Dict:
    """Generate response using configured LLM.
    
    Args:
        state: Current agent state
        llm: LangChain LLM instance
        
    Returns:
        State update with AI response message
    """
    messages = state.get("messages", [])
    
    # Build system prompt with current state
    system_prompt = build_system_prompt(state, include_instructions=True)
    
    # Construct message list with system prompt
    llm_messages = [SystemMessage(content=system_prompt)]
    llm_messages.extend(messages)
    
    # Create tools from loaded skills' scripts (only after paging-in).
    # Backward compatibility: if `loaded_skills` is not present, treat
    # `active_skills` as loaded.
    loaded_skills = state.get("loaded_skills")
    if loaded_skills is None:
        loaded_skills = state.get("active_skills", [])
    skill_loaders = state.get("skill_loaders", {})
    
    # Important: bind skill script tools first so simple fakes/tests that call
    # the first tool behave as expected.
    tools = []
    if loaded_skills and skill_loaders:
        tools.extend(create_tools_from_active_skills(loaded_skills, skill_loaders))
    tools.extend([load_skill, read_skill_resource])
    
    # Bind tools to LLM if available
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    # Generate response
    response = llm_with_tools.invoke(llm_messages)
    
    # Return message to be appended to state
    return {
        "messages": [response]
    }


def should_continue(state: AgentState) -> str:
    """Determine next action in the graph.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name: "tools", "generate", or "end"
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # If the last message has tool calls, execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # Otherwise we're done
    return "end"


def should_generate(state: AgentState) -> str:
    """Route after skill selection.

    If there's no user message, don't call the LLM. This enables "warm" graph
    invocations (e.g., to preload config/skills metadata) without generating.
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"
    return "generate"


def tool_execution_node(state: AgentState) -> Dict:
    """Execute tool calls from the last AI message.
    
    Args:
        state: Current agent state with tool calls in last message
        
    Returns:
        State update with tool results as ToolMessage objects
    """
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    if not isinstance(last_message, AIMessage):
        return {"messages": []}
    
    tool_calls = last_message.tool_calls
    if not tool_calls:
        return {"messages": []}
    
    # Get tools from loaded skills
    active_skills = list(state.get("active_skills", []))
    loaded_skills_val = state.get("loaded_skills")
    loaded_skills = list(loaded_skills_val) if loaded_skills_val is not None else list(active_skills)
    skill_loaders = dict(state.get("skill_loaders", {}))

    tools = []
    if loaded_skills and skill_loaders:
        tools = create_tools_from_active_skills(loaded_skills, skill_loaders)
    
    # Create tool name -> tool mapping
    tools_by_name = {tool.name: tool for tool in tools}
    
    # Execute each tool call
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_id = tool_call["id"]
        
        if tool_name == "load_skill":
            skill_name = None
            if isinstance(tool_args, dict):
                skill_name = tool_args.get("skill_name")
            elif isinstance(tool_args, str):
                skill_name = tool_args

            if not skill_name:
                tool_messages.append(
                    ToolMessage(
                        content="Error: missing skill_name",
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )
                continue

            # Locate skill metadata by name
            skill_meta = None
            for meta in state.get("skills_metadata", []):
                if meta.get("name") == skill_name:
                    skill_meta = meta
                    break

            if not skill_meta:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: unknown skill '{skill_name}'",
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )
                continue

            if skill_name not in active_skills:
                active_skills.append(skill_name)
            if skill_name not in loaded_skills:
                loaded_skills.append(skill_name)
            skill_loaders[skill_name] = skill_meta

            tool_messages.append(
                ToolMessage(
                    content=f"Loaded skill '{skill_name}'",
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

        elif tool_name == "read_skill_resource":
            skill_name = None
            file_path = None
            if isinstance(tool_args, dict):
                skill_name = tool_args.get("skill_name")
                file_path = tool_args.get("file_path")
            if not skill_name or not file_path:
                tool_messages.append(
                    ToolMessage(
                        content="Error: expected skill_name and file_path",
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )
                continue

            # Prefer metadata from loaded skills; fall back to discovered skills.
            meta = skill_loaders.get(skill_name)
            if meta is None:
                for m in state.get("skills_metadata", []):
                    if m.get("name") == skill_name:
                        meta = m
                        break

            if meta is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: unknown skill '{skill_name}'",
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )
                continue

            # Strict sandbox: file must remain within the skill directory
            skill_root = Path(meta["skill_dir"]).resolve()
            candidate = (skill_root / Path(file_path)).resolve()
            try:
                candidate.relative_to(skill_root)
            except Exception as e:
                raise SecurityError(
                    f"Access denied: '{file_path}' is outside skill '{skill_name}'"
                ) from e

            if not candidate.exists() or not candidate.is_file():
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: file not found '{file_path}'",
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )
                continue

            tool_messages.append(
                ToolMessage(
                    content=candidate.read_text(encoding="utf-8"),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

        elif tool_name in tools_by_name:
            try:
                # Invoke the tool
                tool_func = tools_by_name[tool_name]
                result = tool_func.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
            except Exception as e:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing {tool_name}: {e}",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_name} not found",
                    tool_call_id=tool_id,
                    name=tool_name
                )
            )
    
    return {
        "messages": tool_messages,
        "active_skills": active_skills,
        "loaded_skills": loaded_skills,
        "skill_loaders": skill_loaders,
    }


# =============================================================================
# Worker Node Factory (Hierarchical Supervisor Architecture)
# =============================================================================

def build_worker_system_prompt(skill_loader: SkillLoader, state: AgentState) -> str:
    """Build system prompt for a specific worker node.
    
    Args:
        skill_loader: SkillLoader instance for the worker's skill
        state: Current agent state
        
    Returns:
        System prompt string tailored for this worker
    """
    prompt = ""
    
    # Add agent configuration if available (shared context)
    if state.get("agent_config"):
        prompt += state["agent_config"]
        prompt += "\n\n"
    
    # Add this worker's skill instructions
    instructions = skill_loader.load_instructions()
    prompt += f"<active_skill name=\"{skill_loader.name}\">\n"
    prompt += f"Description: {skill_loader.description}\n\n"
    prompt += instructions
    prompt += f"\n</active_skill>\n\n"
    
    # Add skill usage guidance
    prompt += """## Your Role
You are a specialized worker handling tasks related to your skill domain.
Execute the user's request using your skill's capabilities and tools.
When your task is complete, provide a clear response summarizing what was done.

Available resources:
- scripts/: Executable tools you can call
- references/: Documentation you can load
- assets/: Templates and data files

Use the tools available to you to complete the user's request.
"""
    
    return prompt


def create_worker_node(
    skill_name: str,
    llm: BaseChatModel,
    skill_metadata: Dict
) -> Callable[[AgentState], Dict]:
    """Factory function to create a worker node for a specific skill.
    
    This creates a node function that:
    1. Loads the skill's instructions
    2. Binds only that skill's tools (scripts) and standard library tools
    3. Executes LLM generation
    4. Ensures state isolation for internal tool calls
    
    Args:
        skill_name: Name of the skill this worker handles
        llm: LangChain LLM instance
        skill_metadata: Skill metadata dictionary
        
    Returns:
        A callable node function for the LangGraph
    """
    
    def worker_node(state: AgentState) -> Dict:
        """Worker node for {skill_name} skill.
        
        Args:
            state: Current agent state
            
        Returns:
            State update with AI response message
        """
        messages = state.get("messages", [])

        if logger.isEnabledFor(logging.DEBUG):
            human_count = sum(isinstance(m, HumanMessage) for m in messages)
            ai_count = sum(isinstance(m, AIMessage) for m in messages)
            tool_count = sum(isinstance(m, ToolMessage) for m in messages)
            last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            logger.debug(
                "Worker invoked: skill=%s messages=%d (human=%d ai=%d tool=%d) last_user=%r",
                skill_name,
                len(messages),
                human_count,
                ai_count,
                tool_count,
                getattr(last_human, "content", None),
            )
        
        # Create skill loader from metadata
        loader = SkillLoader(skill_metadata)
        
        # Build worker-specific system prompt
        system_prompt = build_worker_system_prompt(loader, state)
        
        # Construct message list with system prompt
        llm_messages = [SystemMessage(content=system_prompt)]
        
        # Filter messages to exclude internal scratchpad from other workers
        # Keep all user messages and final responses, but can filter tool
        # call/response pairs that are internal to other workers if needed
        llm_messages.extend(messages)
        
        # Create tools for this worker
        tools = []
        
        # 1. Add this skill's script tools
        if loader.has_scripts:
            skill_tools = create_tools_from_active_skills([skill_name], {skill_name: skill_metadata})
            tools.extend(skill_tools)
        
        # 2. Add standard library tools if skill requests them
        requested_tools = skill_metadata.get("tools", [])
        if requested_tools:
            standard_tools = get_standard_tools()
            for std_tool in standard_tools:
                if std_tool.name in requested_tools:
                    tools.append(std_tool)
        
        # 3. Add core tools (load_skill, read_skill_resource)
        tools.extend([load_skill, read_skill_resource])
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools) if tools else llm

        if logger.isEnabledFor(logging.DEBUG):
            tool_names = [getattr(t, "name", str(t)) for t in tools]
            logger.debug(
                "Worker tools bound: skill=%s tools=%d names=%s",
                skill_name,
                len(tools),
                tool_names,
            )
        
        # Generate response
        response = llm_with_tools.invoke(llm_messages)

        if logger.isEnabledFor(logging.DEBUG):
            tc = getattr(response, "tool_calls", None)
            logger.debug(
                "Worker LLM response: skill=%s type=%s tool_calls=%s",
                skill_name,
                type(response).__name__,
                (len(tc) if tc else 0),
            )
        
        # Update state to track that this skill is loaded/active
        active_skills = list(state.get("active_skills", []))
        loaded_skills = list(state.get("loaded_skills") or active_skills)
        skill_loaders = dict(state.get("skill_loaders", {}))
        
        if skill_name not in active_skills:
            active_skills.append(skill_name)
        if skill_name not in loaded_skills:
            loaded_skills.append(skill_name)
        skill_loaders[skill_name] = skill_metadata
        
        return {
            "messages": [response],
            "active_skills": active_skills,
            "loaded_skills": loaded_skills,
            "skill_loaders": skill_loaders,
        }
    
    # Set proper function name for debugging
    worker_node.__name__ = f"worker_{skill_name}"
    worker_node.__doc__ = f"Worker node for {skill_name} skill."
    
    return worker_node


def create_worker_tool_execution_node(
    skill_name: str,
    skill_metadata: Dict
) -> Callable[[AgentState], Dict]:
    """Factory function to create a tool execution node for a specific worker.
    
    This handles tool calls made by a specific worker, maintaining state isolation.
    
    Args:
        skill_name: Name of the skill this worker handles
        skill_metadata: Skill metadata dictionary
        
    Returns:
        A callable node function for tool execution
    """
    
    def worker_tool_node(state: AgentState) -> Dict:
        """Execute tool calls for {skill_name} worker.
        
        Args:
            state: Current agent state with tool calls in last message
            
        Returns:
            State update with tool results as ToolMessage objects
        """
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if not isinstance(last_message, AIMessage):
            return {"messages": []}
        
        tool_calls = last_message.tool_calls
        if not tool_calls:
            return {"messages": []}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Worker tools invoked: skill=%s tool_calls=%d names=%s",
                skill_name,
                len(tool_calls),
                [tc.get("name") for tc in tool_calls],
            )
        
        # Create skill loader
        loader = SkillLoader(skill_metadata)
        
        # Get tools for this worker
        tools = []
        if loader.has_scripts:
            skill_tools = create_tools_from_active_skills([skill_name], {skill_name: skill_metadata})
            tools.extend(skill_tools)
        
        # Add standard tools if requested
        requested_tools = skill_metadata.get("tools", [])
        if requested_tools:
            standard_tools = get_standard_tools()
            for std_tool in standard_tools:
                if std_tool.name in requested_tools:
                    tools.append(std_tool)
        
        # Create tool name -> tool mapping
        tools_by_name = {tool.name: tool for tool in tools}
        
        # Track state updates for load_skill and read_skill_resource
        active_skills = list(state.get("active_skills", []))
        loaded_skills = list(state.get("loaded_skills") or active_skills)
        skill_loaders = dict(state.get("skill_loaders", {}))
        
        # Execute each tool call
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call["id"]
            
            # Handle core tools specially
            if tool_name == "load_skill":
                result = _handle_load_skill(
                    tool_args, tool_id, state,
                    active_skills, loaded_skills, skill_loaders
                )
                tool_messages.append(result)
                
            elif tool_name == "read_skill_resource":
                result = _handle_read_skill_resource(
                    tool_args, tool_id, state, skill_loaders
                )
                tool_messages.append(result)
                
            elif tool_name in tools_by_name:
                try:
                    tool_func = tools_by_name[tool_name]
                    result = tool_func.invoke(tool_args)
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id,
                            name=tool_name
                        )
                    )
                except Exception as e:
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error executing {tool_name}: {e}",
                            tool_call_id=tool_id,
                            name=tool_name
                        )
                    )
            else:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool {tool_name} not found",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                )
        
        return {
            "messages": tool_messages,
            "active_skills": active_skills,
            "loaded_skills": loaded_skills,
            "skill_loaders": skill_loaders,
        }
    
    worker_tool_node.__name__ = f"worker_{skill_name}_tools"
    return worker_tool_node


def _handle_load_skill(
    tool_args: Dict,
    tool_id: str,
    state: AgentState,
    active_skills: List[str],
    loaded_skills: List[str],
    skill_loaders: Dict
) -> ToolMessage:
    """Handle load_skill tool call."""
    skill_name = None
    if isinstance(tool_args, dict):
        skill_name = tool_args.get("skill_name")
    elif isinstance(tool_args, str):
        skill_name = tool_args

    if not skill_name:
        return ToolMessage(
            content="Error: missing skill_name",
            tool_call_id=tool_id,
            name="load_skill",
        )

    # Locate skill metadata by name
    skill_meta = None
    for meta in state.get("skills_metadata", []):
        if meta.get("name") == skill_name:
            skill_meta = meta
            break

    if not skill_meta:
        return ToolMessage(
            content=f"Error: unknown skill '{skill_name}'",
            tool_call_id=tool_id,
            name="load_skill",
        )

    if skill_name not in active_skills:
        active_skills.append(skill_name)
    if skill_name not in loaded_skills:
        loaded_skills.append(skill_name)
    skill_loaders[skill_name] = skill_meta

    return ToolMessage(
        content=f"Loaded skill '{skill_name}'",
        tool_call_id=tool_id,
        name="load_skill",
    )


def _handle_read_skill_resource(
    tool_args: Dict,
    tool_id: str,
    state: AgentState,
    skill_loaders: Dict
) -> ToolMessage:
    """Handle read_skill_resource tool call."""
    skill_name = None
    file_path = None
    if isinstance(tool_args, dict):
        skill_name = tool_args.get("skill_name")
        file_path = tool_args.get("file_path")
        
    if not skill_name or not file_path:
        return ToolMessage(
            content="Error: expected skill_name and file_path",
            tool_call_id=tool_id,
            name="read_skill_resource",
        )

    # Prefer metadata from loaded skills; fall back to discovered skills
    meta = skill_loaders.get(skill_name)
    if meta is None:
        for m in state.get("skills_metadata", []):
            if m.get("name") == skill_name:
                meta = m
                break

    if meta is None:
        return ToolMessage(
            content=f"Error: unknown skill '{skill_name}'",
            tool_call_id=tool_id,
            name="read_skill_resource",
        )

    # Strict sandbox: file must remain within the skill directory
    skill_root = Path(meta["skill_dir"]).resolve()
    candidate = (skill_root / Path(file_path)).resolve()
    try:
        candidate.relative_to(skill_root)
    except Exception as e:
        raise SecurityError(
            f"Access denied: '{file_path}' is outside skill '{skill_name}'"
        ) from e

    if not candidate.exists() or not candidate.is_file():
        return ToolMessage(
            content=f"Error: file not found '{file_path}'",
            tool_call_id=tool_id,
            name="read_skill_resource",
        )

    return ToolMessage(
        content=candidate.read_text(encoding="utf-8"),
        tool_call_id=tool_id,
        name="read_skill_resource",
    )


def worker_should_continue(state: AgentState) -> str:
    """Determine if worker should continue with tool execution or return to supervisor.
    
    Args:
        state: Current agent state
        
    Returns:
        "tools" if tool calls pending, "supervisor" otherwise
    """
    messages = state.get("messages", [])
    if not messages:
        return "supervisor"
    
    last_message = messages[-1]
    
    # If the last message has tool calls, execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # Otherwise return to supervisor
    return "supervisor"


