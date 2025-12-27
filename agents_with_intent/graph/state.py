"""State definitions for LangGraph agent."""
from typing import TypedDict, List, Dict, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """State schema for LangGraph agent with progressive skill disclosure.
    
    Attributes:
        messages: Conversation history (LangChain messages)
        skills_metadata: Lightweight skill info for progressive disclosure
        active_skills: Currently activated skill names
        loaded_skills: Skills whose full instructions have been loaded
        skill_loaders: SkillLoader instances for activated skills
        agent_config: Configuration from agent.md
        next_action: Next action to take (for routing) - legacy field
        next: Supervisor routing decision (skill name or "FINISH")
        previous_specialist: Last specialist skill for context preservation
        supervisor_reasoning: Explanation for supervisor's routing decision
    """
    # Conversation history - append-only
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Skill metadata for progressive disclosure (loaded at startup)
    skills_metadata: List[Dict[str, any]]
    
    # Currently activated skills (names)
    active_skills: List[str]

    # Skills whose full instructions have been loaded (progressive disclosure)
    loaded_skills: List[str]
    
    # SkillLoader instances (created when skills are activated)
    skill_loaders: Dict[str, any]  # Dict[str, SkillLoader]
    
    # Agent configuration from agent.md
    agent_config: Optional[str]
    
    # Routing control (legacy - for flat architecture)
    next_action: Optional[str]
    
    # ==========================================================================
    # Hierarchical Supervisor Architecture Fields
    # ==========================================================================
    
    # Supervisor routing decision: skill name to route to, or "FINISH"
    next: Optional[str]
    
    # Previous specialist for context preservation during utility interruptions
    # When a user asks for a utility action (file save, search) while engaged
    # with a specialist, this field tracks which specialist to return to
    previous_specialist: Optional[str]
    
    # Explanation for supervisor's routing decision (for debugging/logging)
    supervisor_reasoning: Optional[str]

