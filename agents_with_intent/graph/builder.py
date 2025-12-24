"""LangGraph graph builder for agent state machine."""
from typing import List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel

from agents_with_intent.graph.state import AgentState
from agents_with_intent.graph.nodes import (
    load_agent_config_node,
    discover_skills_node,
    skill_selection_node,
    llm_generation_node,
    should_continue
)


def create_agent_graph(
    llm: BaseChatModel,
    skills_dirs: List[str],
    agent_config_path: Optional[str] = None
):
    """Create LangGraph state machine for agent with progressive skill discovery.
    
    Graph structure:
    1. START -> load_config -> discover_skills -> skill_selection -> generate -> END
    2. Loop back from END to skill_selection for multi-turn conversations
    
    Args:
        llm: LangChain LLM instance (any compatible model)
        skills_dirs: List of directories to scan for skills
        agent_config_path: Optional path to agent.md configuration file
        
    Returns:
        Compiled LangGraph graph with checkpointing
    """
    # Create state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node(
        "load_config",
        lambda state: load_agent_config_node(state, agent_config_path)
    )
    
    workflow.add_node(
        "discover_skills",
        lambda state: discover_skills_node(state, skills_dirs)
    )
    
    workflow.add_node(
        "skill_selection",
        skill_selection_node
    )
    
    workflow.add_node(
        "generate",
        lambda state: llm_generation_node(state, llm)
    )
    
    # Define edges
    workflow.set_entry_point("load_config")
    
    workflow.add_edge("load_config", "discover_skills")
    workflow.add_edge("discover_skills", "skill_selection")
    workflow.add_edge("skill_selection", "generate")
    workflow.add_edge("generate", END)
    
    # Add checkpointing for conversation state
    checkpointer = MemorySaver()
    
    # Compile graph
    return workflow.compile(checkpointer=checkpointer)
