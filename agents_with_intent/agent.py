"""Main Agent class for agents-with-intent library."""
from typing import List, Optional, Dict
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from agents_with_intent.graph.builder import create_agent_graph
from agents_with_intent.graph.state import AgentState


class Agent:
    """AI Agent with progressive skill discovery following Agent Skills spec.
    
    This agent uses LangGraph for state management and supports:
    - Progressive skill discovery (metadata first, content on-demand)
    - Agent Skills specification compliance
    - Any LangChain-compatible LLM
    - Script execution, reference loading, asset access
    - Multi-turn conversations with checkpointing
    
    Example:
        ```python
        from agents_with_intent import Agent
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        agent = Agent(
            llm=llm,
            skills_dirs=["./skills"],
            agent_config_path="./agent.md"
        )
        
        response = agent.run("Help me with my budget")
        print(response)
        
        # Continue conversation
        response = agent.run("What about retirement savings?")
        ```
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        skills_dirs: Optional[List[str]] = None,
        agent_config_path: Optional[str] = None,
        thread_id: str = "default",
        eager_init: bool = True,
    ):
        """Initialize agent with LLM and skill configuration.
        
        Args:
            llm: LangChain LLM instance (pre-configured)
                 Examples:
                 - ChatOpenAI(model="gpt-4")
                 - ChatAnthropic(model="claude-3-5-sonnet-20241022")
                 - ChatOpenAI(base_url="http://localhost:11434/v1", model="llama3.1")
            skills_dirs: List of directories to scan for SKILL.md files
            agent_config_path: Optional path to agent.md configuration file
            thread_id: Thread ID for conversation checkpointing (default: "default")
            eager_init: If True (default), run a "warm" graph invocation to
                        populate agent config and skills metadata. This is safe
                        and should not trigger an LLM call.
        """
        self.llm = llm
        self.skills_dirs = skills_dirs or ["./skills"]
        self.agent_config_path = agent_config_path
        self.thread_id = thread_id
        
        # Validate skills directories
        for skills_dir in self.skills_dirs:
            skills_path = Path(skills_dir)
            if not skills_path.exists():
                raise ValueError(f"Skills directory not found: {skills_dir}")
            if not skills_path.is_dir():
                raise ValueError(f"Skills path is not a directory: {skills_dir}")
        
        # Validate agent config if provided
        if agent_config_path:
            config_path = Path(agent_config_path)
            if not config_path.exists():
                raise ValueError(f"Agent config file not found: {agent_config_path}")
        
        # Create LangGraph state machine
        self.graph = create_agent_graph(
            llm=llm,
            skills_dirs=self.skills_dirs,
            agent_config_path=agent_config_path
        )
        
        # Initialize state for this thread by running initial discovery
        self._state: Optional[AgentState] = None
        if eager_init:
            try:
                config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 100}
                # Run the graph once to populate skills metadata and agent config
                self.graph.invoke({}, config=config)
            except Exception:
                # Don't fail agent construction if discovery has issues; allow later invocation
                pass
    
    def run(self, user_input: str) -> str:
        """Run agent with user input and return response.
        
        This method handles a single turn of conversation.
        State is preserved across calls via checkpointing.
        
        Args:
            user_input: User's message/query
            
        Returns:
            Agent's response as string
        """
        # Create config with thread ID for checkpointing
        config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 100}
        
        # Prepare input with user message
        input_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        # Run graph
        result = self.graph.invoke(input_state, config=config)
        
        # Extract AI response from messages
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
        
        return "I apologize, but I couldn't generate a response."
    
    def stream(self, user_input: str):
        """Stream agent responses token-by-token.
        
        Args:
            user_input: User's message/query
            
        Yields:
            Response tokens as they're generated
        """
        # LangGraph streams node updates keyed by node name, e.g.
        # {"generate": {"messages": [AIMessage(...)]}}.
        config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 100}

        input_state = {
            "messages": [HumanMessage(content=user_input)]
        }

        # Stream graph execution (node-by-node updates)
        for update in self.graph.stream(input_state, config=config, stream_mode="updates"):
            if not isinstance(update, dict):
                continue

            for _node_name, chunk in update.items():
                if not isinstance(chunk, dict):
                    continue

                messages = chunk.get("messages")
                if not messages:
                    continue

                for message in messages:
                    if isinstance(message, AIMessage):
                        yield message.content
    
    def get_state(self) -> Dict:
        """Get current agent state.
        
        Returns:
            Current state dictionary with:
            - messages: Conversation history
            - skills_metadata: Discovered skills
            - active_skills: Currently activated skill names
            - agent_config: Agent configuration
        """
        config = {"configurable": {"thread_id": self.thread_id}}
        state_snapshot = self.graph.get_state(config)
        return state_snapshot.values if state_snapshot else {}
    
    def reset(self):
        """Reset agent state and start a new conversation.
        
        This clears conversation history but keeps skill metadata.
        """
        # Create a new thread ID
        import time
        self.thread_id = f"thread_{int(time.time() * 1000)}"
    
    def list_skills(self) -> List[Dict[str, str]]:
        """List all discovered skills with metadata.
        
        Returns:
            List of dictionaries with 'name' and 'description' keys
        """
        state = self.get_state()
        skills_metadata = state.get("skills_metadata", [])
        
        return [
            {
                "name": skill["name"],
                "description": skill["description"],
                "has_scripts": skill.get("has_scripts", False),
                "has_references": skill.get("has_references", False),
                "has_assets": skill.get("has_assets", False),
            }
            for skill in skills_metadata
        ]
    
    def get_active_skills(self) -> List[str]:
        """Get list of currently activated skill names.
        
        Returns:
            List of skill names that have been activated
        """
        state = self.get_state()
        return state.get("active_skills", [])
    
    def interactive(self):
        """Start an interactive chat session with the agent.
        
        Type 'exit', 'quit', or press Ctrl+C to stop.
        Type 'skills' to list available skills.
        Type 'active' to show activated skills.
        Type 'reset' to start a new conversation.
        """
        print("🤖 Agent Interactive Mode")
        print("=" * 60)
        print("Commands: 'skills', 'active', 'reset', 'exit'")
        print("=" * 60)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit"]:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == "skills":
                    skills = self.list_skills()
                    if skills:
                        print("\n📚 Available Skills:")
                        for skill in skills:
                            print(f"  • {skill['name']}: {skill['description']}")
                            flags = []
                            if skill['has_scripts']:
                                flags.append("scripts")
                            if skill['has_references']:
                                flags.append("references")
                            if skill['has_assets']:
                                flags.append("assets")
                            if flags:
                                print(f"    Resources: {', '.join(flags)}")
                    else:
                        print("\n📚 No skills discovered")
                    print()
                    continue
                
                if user_input.lower() == "active":
                    active = self.get_active_skills()
                    if active:
                        print(f"\n✅ Active Skills: {', '.join(active)}")
                    else:
                        print("\n✅ No skills currently activated")
                    print()
                    continue
                
                if user_input.lower() == "reset":
                    self.reset()
                    print("\n🔄 Conversation reset\n")
                    continue
                
                # Generate response
                print("\nAgent: ", end="", flush=True)
                response = self.run(user_input)
                print(response)
                print()
                
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import os
                if os.environ.get("DEBUG"):
                    import traceback
                    traceback.print_exc()
                print()
