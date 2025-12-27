"""LangGraph graph builder for agent state machine.

This module provides two graph architectures:
1. create_agent_graph() - Original flat accumulator architecture (Select -> Generate -> Tools)
2. create_supervisor_graph() - Hierarchical supervisor architecture (Supervisor -> Workers)

The supervisor architecture is inspired by LangGraph Supervisor pattern and provides:
- Better handling of multi-turn workflows
- Clear distinction between "Chat Agents" (long-running) and "Task Tools" (short-lived)
- Context preservation across skill switches
"""
from typing import List, Optional, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel

from agents_with_intent.graph.state import AgentState
from agents_with_intent.graph.nodes import (
    load_agent_config_node,
    discover_skills_node,
    skill_selection_node,
    llm_generation_node,
    tool_execution_node,
    should_continue,
    should_generate,
    create_worker_node,
    create_worker_tool_execution_node,
    worker_should_continue,
)
from agents_with_intent.graph.supervisor import (
    supervisor_node,
    route_supervisor,
)
from agents_with_intent.skills.discovery import discover_skills


def create_agent_graph(
    llm: BaseChatModel,
    skills_dirs: List[str],
    agent_config_path: Optional[str] = None
):
    """Create LangGraph state machine for agent with progressive skill discovery.
    
    This is the original "flat accumulator" architecture:
    1. START -> load_config -> discover_skills -> skill_selection -> generate
    2. After generate: check if tool calls exist
       - If yes: execute tools -> generate again (with tool results)
       - If no: END
    
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
    
    workflow.add_node(
        "tools",
        tool_execution_node
    )
    
    # Define edges
    workflow.set_entry_point("load_config")
    
    workflow.add_edge("load_config", "discover_skills")
    workflow.add_edge("discover_skills", "skill_selection")

    # If there is no user input yet, don't call the LLM.
    workflow.add_conditional_edges(
        "skill_selection",
        should_generate,
        {
            "generate": "generate",
            "end": END,
        },
    )
    
    # Conditional edge after generate: check for tool calls
    workflow.add_conditional_edges(
        "generate",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After executing tools, generate response with tool results
    workflow.add_edge("tools", "generate")
    
    # Add checkpointing for conversation state
    checkpointer = MemorySaver()
    
    # Compile graph
    return workflow.compile(checkpointer=checkpointer)


def create_supervisor_graph(
    llm: BaseChatModel,
    skills_dirs: List[str],
    agent_config_path: Optional[str] = None,
    skills_metadata: Optional[List[Dict]] = None
):
    """Create hierarchical supervisor graph for multi-turn agent workflows.
    
    This architecture uses a Supervisor -> Worker model:
    1. START -> load_config -> discover_skills -> supervisor
    2. Supervisor decides which worker (skill) should handle the request
    3. Worker executes task, then returns to supervisor
    4. Supervisor can route to another worker or FINISH
    
    Graph structure:
    ```
                    +---> worker_skill_A ---> tools_A ---+
                    |          ^                  |      |
    START -> init --+          +------------------+      |
                    |                                    |
                    +---> supervisor <-------------------+
                    |          |
                    +---> worker_skill_B ---> tools_B ---+
                    |          ^                  |      |
                    |          +------------------+      |
                    |                                    |
                    +---> FINISH/END <------------------+
    ```
    
    Args:
        llm: LangChain LLM instance (any compatible model)
        skills_dirs: List of directories to scan for skills
        agent_config_path: Optional path to agent.md configuration file
        skills_metadata: Pre-discovered skills metadata (optional, for testing)
        
    Returns:
        Compiled LangGraph graph with checkpointing
    """
    # Discover skills upfront to build the graph dynamically
    if skills_metadata is None:
        skills_metadata = discover_skills(skills_dirs)
    
    # Create state graph
    workflow = StateGraph(AgentState)
    
    # ==========================================================================
    # Initialization Nodes
    # ==========================================================================
    
    workflow.add_node(
        "load_config",
        lambda state: load_agent_config_node(state, agent_config_path)
    )
    
    workflow.add_node(
        "discover_skills",
        lambda state: discover_skills_node(state, skills_dirs)
    )
    
    # ==========================================================================
    # Supervisor Node
    # ==========================================================================
    
    workflow.add_node(
        "supervisor",
        lambda state: supervisor_node(state, llm)
    )
    
    # ==========================================================================
    # Worker Nodes (one per discovered skill)
    # ==========================================================================
    
    worker_nodes = []
    for skill_meta in skills_metadata:
        skill_name = skill_meta["name"]
        
        # Create worker node for this skill
        worker_node_fn = create_worker_node(skill_name, llm, skill_meta)
        workflow.add_node(skill_name, worker_node_fn)
        
        # Create tool execution node for this skill
        tool_node_name = f"{skill_name}_tools"
        tool_node_fn = create_worker_tool_execution_node(skill_name, skill_meta)
        workflow.add_node(tool_node_name, tool_node_fn)
        
        worker_nodes.append(skill_name)
    
    # ==========================================================================
    # Entry Point and Initialization Edges
    # ==========================================================================
    
    workflow.set_entry_point("load_config")
    workflow.add_edge("load_config", "discover_skills")
    workflow.add_edge("discover_skills", "supervisor")
    
    # ==========================================================================
    # Supervisor Routing (Conditional Edges)
    # ==========================================================================
    
    # Build routing map: skill_name -> skill_name, "end" -> END
    routing_map = {"end": END}
    for skill_name in worker_nodes:
        routing_map[skill_name] = skill_name
    
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        routing_map
    )
    
    # ==========================================================================
    # Worker -> Tools -> Worker -> Supervisor Edges
    # ==========================================================================
    
    for skill_name in worker_nodes:
        tool_node_name = f"{skill_name}_tools"
        
        # Create a closure to capture the correct tool_node_name
        def create_worker_router(tn):
            def router(state):
                result = worker_should_continue(state)
                if result == "tools":
                    return "tools"
                return "supervisor"
            return router
        
        # After worker generates: check for tool calls or return to supervisor
        workflow.add_conditional_edges(
            skill_name,
            create_worker_router(tool_node_name),
            {
                "tools": tool_node_name,
                "supervisor": "supervisor"
            }
        )
        
        # After tool execution: return to worker for more generation
        workflow.add_edge(tool_node_name, skill_name)
    
    # ==========================================================================
    # Compile with Checkpointing
    # ==========================================================================
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Alias for backward compatibility
create_graph = create_agent_graph

