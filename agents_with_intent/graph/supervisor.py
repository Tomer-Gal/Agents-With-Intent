"""Supervisor node for hierarchical agent architecture.

The supervisor decides which worker (skill) should handle the current request,
or whether to finish and respond to the user.
"""
from typing import Dict, List, Optional
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from agents_with_intent.graph.state import AgentState


logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    """Structured output for supervisor routing decision."""
    next: str = Field(
        description="The next worker to route to, or 'FINISH' to respond to the user"
    )
    reasoning: str = Field(
        description="Brief explanation for the routing decision"
    )


SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor managing a team of specialized workers. Your job is to analyze user requests and route them to the appropriate worker, or respond directly when appropriate.

## Your Role
- Analyze each user message to determine which worker should handle it
- Route to specialists for domain-specific tasks
- Route to utility workers for quick actions (file operations, calculations, searches)
- Respond with FINISH when the conversation is complete or when you can answer directly

## Available Workers
{workers_list}

## Routing Guidelines

1. **Primary Specialist Routing**: When a user's request clearly matches a worker's domain, route to that worker.

2. **Utility Interruptions**: When a user asks for a utility action (like saving a file, searching, or calculation) while engaged with a primary specialist, route to the utility skill. Treat this as a temporary interruption. Once the utility skill completes its task, you must default to routing back to the previous specialist to continue the conversation context.

3. **Context Preservation**: Track which specialist the user was engaged with before a utility interruption. After the utility task completes, route back to that specialist.

4. **Explicit Topic Changes**: If the user explicitly requests to change topics or start a new task unrelated to the current specialist, route to the appropriate new specialist.

5. **Finish Conditions**: Route to FINISH when:
   - The user's request has been fully addressed
   - The user says goodbye or indicates they're done
   - You can provide a simple response without specialist help
   - No worker matches the request and it's a general query

## Current Conversation Context
Previous specialist (if any): {previous_specialist}

## Response Format
You must respond with a JSON object containing:
- "next": The worker name to route to, or "FINISH"
- "reasoning": Brief explanation for your decision

Only respond with valid worker names from the list above, or "FINISH"."""


def build_workers_list(skills_metadata: List[Dict]) -> str:
    """Build a formatted list of available workers from skill metadata.
    
    Args:
        skills_metadata: List of skill metadata dictionaries
        
    Returns:
        Formatted string listing all workers and their descriptions
    """
    if not skills_metadata:
        return "No specialized workers available."
    
    lines = []
    for skill in skills_metadata:
        name = skill.get("name", "unknown")
        description = skill.get("description", "No description available")
        lines.append(f"- **{name}**: {description}")
    
    return "\n".join(lines)


def get_previous_specialist(state: AgentState) -> Optional[str]:
    """Extract the previous specialist from state or message history.
    
    Args:
        state: Current agent state
        
    Returns:
        Name of previous specialist or None
    """
    return state.get("previous_specialist")


def supervisor_node(state: AgentState, llm: BaseChatModel) -> Dict:
    """Supervisor node that decides routing based on user input and context.
    
    The supervisor analyzes the conversation and decides which worker should
    handle the request, or whether to finish.
    
    Args:
        state: Current agent state
        llm: LangChain LLM instance
        
    Returns:
        State update with 'next' field indicating routing decision
    """
    messages = state.get("messages", [])
    skills_metadata = state.get("skills_metadata", [])

    if logger.isEnabledFor(logging.DEBUG):
        human_count = sum(isinstance(m, HumanMessage) for m in messages)
        ai_count = sum(isinstance(m, AIMessage) for m in messages)
        system_count = sum(isinstance(m, SystemMessage) for m in messages)
        last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        logger.debug(
            "Supervisor invoked: messages=%d (human=%d ai=%d system=%d) skills=%d previous_specialist=%s last_user=%r",
            len(messages),
            human_count,
            ai_count,
            system_count,
            len(skills_metadata),
            state.get("previous_specialist"),
            getattr(last_human, "content", None),
        )
    
    # Build workers list from skills metadata
    workers_list = build_workers_list(skills_metadata)
    
    # Get previous specialist for context preservation
    previous_specialist = get_previous_specialist(state) or "None"
    
    # Build system prompt
    system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
        workers_list=workers_list,
        previous_specialist=previous_specialist
    )
    
    # Construct messages for supervisor
    llm_messages = [SystemMessage(content=system_prompt)]
    llm_messages.extend(messages)
    
    # Add routing instruction
    llm_messages.append(HumanMessage(content="""Based on the conversation above, decide the next step.
Respond with a JSON object: {"next": "<worker_name or FINISH>", "reasoning": "<brief explanation>"}"""))
    
    # Try to use structured output if available
    try:
        structured_llm = llm.with_structured_output(SupervisorDecision)
        response = structured_llm.invoke(llm_messages)
        next_worker = response.next
        reasoning = response.reasoning
    except (AttributeError, NotImplementedError):
        # Fallback: parse JSON from regular response
        response = llm.invoke(llm_messages)
        next_worker, reasoning = _parse_supervisor_response(response.content, skills_metadata)
    
    # Validate the decision
    valid_workers = [s["name"] for s in skills_metadata] + ["FINISH"]
    if next_worker not in valid_workers:
        # Default to FINISH if invalid worker
        invalid = next_worker
        next_worker = "FINISH"
        reasoning = f"Invalid worker '{invalid}' specified, defaulting to FINISH"
    
    # Track specialist for context preservation (non-utility workers)
    # A worker is considered "utility" if it's a quick task, otherwise it's a specialist
    new_previous_specialist = state.get("previous_specialist")
    if next_worker != "FINISH":
        # Update previous specialist if routing to a non-utility skill
        # For simplicity, treat the routed skill as the new specialist
        # More sophisticated logic could check skill metadata for "utility" flag
        new_previous_specialist = next_worker

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Supervisor decision: next=%s previous_specialist=%s reasoning=%r",
            next_worker,
            new_previous_specialist,
            reasoning,
        )
    
    return {
        "next": next_worker,
        "previous_specialist": new_previous_specialist,
        "supervisor_reasoning": reasoning
    }


def _parse_supervisor_response(content: str, skills_metadata: List[Dict]) -> tuple:
    """Parse supervisor response when structured output is not available.
    
    Args:
        content: Raw LLM response content
        skills_metadata: List of skill metadata for validation
        
    Returns:
        Tuple of (next_worker, reasoning)
    """
    import json
    import re
    
    # Try to extract JSON from response
    try:
        # Look for JSON object in response
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            next_worker = data.get("next", "FINISH")
            reasoning = data.get("reasoning", "No reasoning provided")
            return next_worker, reasoning
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Fallback: look for skill names or FINISH in content
    valid_workers = [s["name"] for s in skills_metadata]
    
    content_lower = content.lower()
    if "finish" in content_lower:
        return "FINISH", "Parsed FINISH from response"
    
    for worker in valid_workers:
        if worker.lower() in content_lower:
            return worker, f"Parsed {worker} from response"
    
    return "FINISH", "Could not parse response, defaulting to FINISH"


def route_supervisor(state: AgentState) -> str:
    """Route based on supervisor decision.
    
    This function is used as a conditional edge in the graph.
    
    Args:
        state: Current agent state with 'next' field set by supervisor
        
    Returns:
        Next node name (skill name or "end")
    """
    next_node = state.get("next", "FINISH")
    
    if next_node == "FINISH":
        return "end"
    
    # Return the skill name as the node to route to
    return next_node
