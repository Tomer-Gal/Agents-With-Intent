"""Graph nodes for LangGraph agent state machine."""
from typing import Dict, List, Optional
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from agents_with_intent.graph.state import AgentState
from agents_with_intent.skills.discovery import discover_skills
from agents_with_intent.skills.loader import SkillLoader


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
    
    # Add available skills
    skills_metadata = state.get("skills_metadata", [])
    if skills_metadata:
        prompt += "<available_skills>\n"
        for skill_meta in skills_metadata:
            # Create temporary loader for prompt generation
            loader = SkillLoader(skill_meta)
            prompt += loader.to_prompt_context(include_instructions=False)
            prompt += "\n"
        prompt += "</available_skills>\n\n"
    
    # Add activated skill instructions
    active_skills = state.get("active_skills", [])
    skill_loaders = state.get("skill_loaders", {})
    if active_skills and skill_loaders:
        prompt += "<activated_skills>\n"
        for skill_name in active_skills:
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
- scripts/: Executable code (call when needed)
- references/: Detailed documentation (load when needed)
- assets/: Templates, data files, images (access when needed)
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
    
    # Simple keyword matching (TODO: use semantic similarity in production)
    for skill_meta in skills_metadata:
        skill_name = skill_meta['name']
        description = skill_meta['description'].lower()
        
        # Skip if already active
        if skill_name in active_skills:
            continue
        
        # Check if any description words appear in user message
        # This is a simple heuristic - production should use embeddings
        desc_words = set(description.split())
        msg_words = set(last_message.split())
        
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
    
    # Generate response
    response = llm.invoke(llm_messages)
    
    # Return message to be appended to state
    return {
        "messages": [response]
    }


def should_continue(state: AgentState) -> str:
    """Determine next action in the graph.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name: "generate", "end"
    """
    # Simple routing - always generate for now
    # Could add more sophisticated routing logic here
    return "generate"
