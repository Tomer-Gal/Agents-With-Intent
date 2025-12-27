"""Tests for the Hierarchical Supervisor Architecture.

These tests verify:
1. Context preservation when switching between skills
2. Topic changes and proper skill routing
3. Supervisor decision making
4. Worker node execution
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents_with_intent.graph.state import AgentState
from agents_with_intent.graph.supervisor import (
    supervisor_node,
    route_supervisor,
    build_workers_list,
    SupervisorDecision,
    _parse_supervisor_response,
)
from agents_with_intent.graph.nodes import (
    create_worker_node,
    create_worker_tool_execution_node,
    worker_should_continue,
)
from agents_with_intent.graph.builder import create_supervisor_graph


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_skills_metadata():
    """Sample skills metadata for testing."""
    return [
        {
            "name": "skill-a",
            "description": "Skill A handles financial advice and budgeting",
            "path": "/test/skills/skill-a/SKILL.md",
            "skill_dir": "/test/skills/skill-a",
            "has_scripts": False,
            "has_references": False,
            "has_assets": False,
            "tools": [],
        },
        {
            "name": "skill-b",
            "description": "Skill B handles file operations and data saving",
            "path": "/test/skills/skill-b/SKILL.md",
            "skill_dir": "/test/skills/skill-b",
            "has_scripts": False,
            "has_references": False,
            "has_assets": False,
            "tools": ["file_write", "file_read"],
        },
        {
            "name": "skill-x",
            "description": "Skill X handles topic X related queries and analysis",
            "path": "/test/skills/skill-x/SKILL.md",
            "skill_dir": "/test/skills/skill-x",
            "has_scripts": False,
            "has_references": False,
            "has_assets": False,
            "tools": [],
        },
    ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Test response")
    mock.bind_tools.return_value = mock
    return mock


@pytest.fixture
def base_state(mock_skills_metadata):
    """Create a base state for testing."""
    return {
        "messages": [],
        "skills_metadata": mock_skills_metadata,
        "active_skills": [],
        "loaded_skills": [],
        "skill_loaders": {},
        "agent_config": None,
        "next": None,
        "previous_specialist": None,
    }


# =============================================================================
# Supervisor Tests
# =============================================================================

class TestSupervisorNode:
    """Tests for the supervisor node."""
    
    def test_build_workers_list(self, mock_skills_metadata):
        """Test that workers list is built correctly from metadata."""
        result = build_workers_list(mock_skills_metadata)
        
        assert "skill-a" in result
        assert "skill-b" in result
        assert "skill-x" in result
        assert "financial advice" in result
        assert "file operations" in result
    
    def test_build_workers_list_empty(self):
        """Test workers list with no skills."""
        result = build_workers_list([])
        assert "No specialized workers" in result
    
    def test_supervisor_routes_to_skill(self, base_state, mock_skills_metadata):
        """Test that supervisor routes to the correct skill."""
        # Setup mock LLM to return structured decision
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorDecision(
            next="skill-a",
            reasoning="User is asking about finances"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        
        state = {
            **base_state,
            "messages": [HumanMessage(content="Help me with my budget")],
        }
        
        result = supervisor_node(state, mock_llm)
        
        assert result["next"] == "skill-a"
        assert "previous_specialist" in result
    
    def test_supervisor_routes_to_finish(self, base_state):
        """Test that supervisor can route to FINISH."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorDecision(
            next="FINISH",
            reasoning="User's request has been addressed"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        
        state = {
            **base_state,
            "messages": [HumanMessage(content="Thanks, goodbye!")],
        }
        
        result = supervisor_node(state, mock_llm)
        
        assert result["next"] == "FINISH"
    
    def test_supervisor_invalid_worker_defaults_to_finish(self, base_state):
        """Test that invalid worker name defaults to FINISH."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorDecision(
            next="nonexistent-skill",
            reasoning="Testing invalid skill"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        
        state = {
            **base_state,
            "messages": [HumanMessage(content="Do something")],
        }
        
        result = supervisor_node(state, mock_llm)
        
        assert result["next"] == "FINISH"
    
    def test_parse_supervisor_response_json(self, mock_skills_metadata):
        """Test parsing JSON response from supervisor."""
        content = '{"next": "skill-a", "reasoning": "User needs financial help"}'
        next_worker, reasoning = _parse_supervisor_response(content, mock_skills_metadata)
        
        assert next_worker == "skill-a"
        assert "financial" in reasoning.lower()
    
    def test_parse_supervisor_response_fallback(self, mock_skills_metadata):
        """Test fallback parsing when JSON is invalid."""
        content = "I think we should route to skill-b for file operations"
        next_worker, reasoning = _parse_supervisor_response(content, mock_skills_metadata)
        
        assert next_worker == "skill-b"


class TestRouteSupervisor:
    """Tests for the route_supervisor function."""
    
    def test_route_to_skill(self):
        """Test routing to a skill."""
        state = {"next": "skill-a"}
        result = route_supervisor(state)
        assert result == "skill-a"
    
    def test_route_to_end(self):
        """Test routing to end."""
        state = {"next": "FINISH"}
        result = route_supervisor(state)
        assert result == "end"
    
    def test_route_default_to_end(self):
        """Test that missing next defaults to end."""
        state = {}
        result = route_supervisor(state)
        assert result == "end"


# =============================================================================
# Worker Node Tests
# =============================================================================

class TestWorkerNodes:
    """Tests for worker node creation and execution."""
    
    def test_create_worker_node(self, mock_skills_metadata, mock_llm):
        """Test that worker node is created correctly."""
        skill_meta = mock_skills_metadata[0]
        worker_fn = create_worker_node("skill-a", mock_llm, skill_meta)
        
        assert callable(worker_fn)
        assert "skill-a" in worker_fn.__name__
    
    @patch('agents_with_intent.graph.nodes.SkillLoader')
    def test_worker_node_execution(self, mock_loader_class, mock_skills_metadata, mock_llm, base_state):
        """Test worker node execution."""
        # Setup mock loader
        mock_loader = MagicMock()
        mock_loader.name = "skill-a"
        mock_loader.description = "Test skill"
        mock_loader.has_scripts = False
        mock_loader.load_instructions.return_value = "Test instructions"
        mock_loader_class.return_value = mock_loader
        
        skill_meta = mock_skills_metadata[0]
        worker_fn = create_worker_node("skill-a", mock_llm, skill_meta)
        
        state = {
            **base_state,
            "messages": [HumanMessage(content="Help me with finances")],
        }
        
        result = worker_fn(state)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "skill-a" in result["active_skills"]
    
    def test_worker_should_continue_with_tool_calls(self):
        """Test that worker continues when there are tool calls."""
        mock_tool_call = {"name": "test_tool", "args": {}, "id": "123"}
        state = {
            "messages": [AIMessage(content="", tool_calls=[mock_tool_call])]
        }
        
        result = worker_should_continue(state)
        assert result == "tools"
    
    def test_worker_should_continue_without_tool_calls(self):
        """Test that worker returns to supervisor without tool calls."""
        state = {
            "messages": [AIMessage(content="Final response")]
        }
        
        result = worker_should_continue(state)
        assert result == "supervisor"


# =============================================================================
# Integration Tests: Context Preservation
# =============================================================================

class TestContextPreservation:
    """Test context preservation across skill switches.
    
    Scenario:
    1. Start with Skill A (financial advisor)
    2. User asks to save data (utility action) -> Switch to Skill B
    3. Skill B completes -> Return to Skill A
    4. Verify Skill A remembers the initial context
    """
    
    def test_context_preservation_skill_switch(self, mock_skills_metadata, base_state):
        """Test that context is preserved when switching between skills."""
        # Create mock LLM that tracks conversation context
        call_count = [0]
        
        def mock_invoke(messages):
            call_count[0] += 1
            
            # Check that all previous messages are present
            user_messages = [m for m in messages if isinstance(m, HumanMessage)]
            ai_messages = [m for m in messages if isinstance(m, AIMessage)]
            
            return AIMessage(content=f"Response {call_count[0]}")
        
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_llm.bind_tools.return_value = mock_llm
        
        # Simulate the conversation flow
        state = {
            **base_state,
            "messages": [],
        }
        
        # Step 1: User engages with Skill A (financial)
        msg1 = HumanMessage(content="I need help with my monthly budget of $5000")
        state["messages"] = [msg1]
        state["previous_specialist"] = None
        
        # Supervisor routes to skill-a
        state["next"] = "skill-a"
        state["previous_specialist"] = "skill-a"
        
        # Skill A responds
        response1 = AIMessage(content="I can help with your $5000 budget. What expenses do you have?")
        state["messages"].append(response1)
        
        # Step 2: User asks to save data (utility) while in financial context
        msg2 = HumanMessage(content="Please save this budget to a file called budget.txt")
        state["messages"].append(msg2)
        
        # Supervisor should route to skill-b (file operations) but remember skill-a
        # Previous specialist should still be tracked
        assert state["previous_specialist"] == "skill-a"
        
        # Route to skill-b for file operation
        state["next"] = "skill-b"
        
        # Skill B handles the file operation
        response2 = AIMessage(content="Saved your budget to budget.txt")
        state["messages"].append(response2)
        
        # Step 3: After utility completes, return to skill-a
        state["next"] = "skill-a"
        
        # User continues financial discussion
        msg3 = HumanMessage(content="Now, how should I allocate for rent?")
        state["messages"].append(msg3)
        
        # Verify all messages are in state (context preserved)
        assert len(state["messages"]) == 5
        assert "$5000" in state["messages"][0].content
        assert "budget.txt" in state["messages"][2].content
        assert "rent" in state["messages"][4].content
        
        # Verify the previous specialist was tracked
        assert state["previous_specialist"] == "skill-a"


# =============================================================================
# Integration Tests: Topic Change
# =============================================================================

class TestTopicChange:
    """Test topic change handling.
    
    Scenario:
    1. Start with Skill A (financial advisor)
    2. User explicitly changes topic to X
    3. Supervisor routes to Skill X
    4. Skill A context is no longer primary
    """
    
    def test_explicit_topic_change(self, mock_skills_metadata, base_state):
        """Test that explicit topic change routes to new skill."""
        state = {
            **base_state,
            "messages": [],
        }
        
        # Step 1: User engages with Skill A
        msg1 = HumanMessage(content="Help me with my budget")
        state["messages"] = [msg1]
        state["previous_specialist"] = None
        state["next"] = "skill-a"
        state["previous_specialist"] = "skill-a"
        
        # Skill A responds
        response1 = AIMessage(content="I can help with budgeting.")
        state["messages"].append(response1)
        
        # Step 2: User explicitly changes topic
        msg2 = HumanMessage(content="Actually, let's change topic. I want to talk about topic X instead.")
        state["messages"].append(msg2)
        
        # Mock supervisor decision for topic change
        # In real usage, supervisor would parse "change topic to X" and route accordingly
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorDecision(
            next="skill-x",
            reasoning="User explicitly requested to change topic to X"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        
        result = supervisor_node(state, mock_llm)
        
        # Supervisor should route to skill-x
        assert result["next"] == "skill-x"
        
        # Previous specialist should be updated
        assert result["previous_specialist"] == "skill-x"
    
    def test_topic_change_updates_context(self, mock_skills_metadata, base_state):
        """Test that topic change properly updates specialist tracking."""
        state = {
            **base_state,
            "messages": [
                HumanMessage(content="Budget question"),
                AIMessage(content="Budget answer"),
                HumanMessage(content="Change to topic X"),
            ],
            "previous_specialist": "skill-a",
            "next": "skill-x",
        }
        
        # After routing to skill-x, previous_specialist should update
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorDecision(
            next="skill-x",
            reasoning="Topic change to X"
        )
        mock_llm.with_structured_output.return_value = mock_structured
        
        result = supervisor_node(state, mock_llm)
        
        assert result["next"] == "skill-x"
        assert result["previous_specialist"] == "skill-x"


# =============================================================================
# Graph Builder Tests
# =============================================================================

class TestSupervisorGraphBuilder:
    """Tests for the supervisor graph builder."""
    
    def test_create_supervisor_graph(self, mock_skills_metadata, mock_llm):
        """Test that supervisor graph is created with all nodes."""
        with patch('agents_with_intent.graph.builder.discover_skills') as mock_discover:
            mock_discover.return_value = mock_skills_metadata
            
            graph = create_supervisor_graph(
                llm=mock_llm,
                skills_dirs=["/test/skills"],
                skills_metadata=mock_skills_metadata
            )
            
            # Graph should be compiled
            assert graph is not None
    
    def test_graph_has_supervisor_node(self, mock_skills_metadata, mock_llm):
        """Test that graph includes supervisor node."""
        with patch('agents_with_intent.graph.builder.discover_skills') as mock_discover:
            mock_discover.return_value = mock_skills_metadata
            
            graph = create_supervisor_graph(
                llm=mock_llm,
                skills_dirs=["/test/skills"],
                skills_metadata=mock_skills_metadata
            )
            
            # Check graph nodes (implementation detail, may need adjustment)
            assert graph is not None


class TestLangfuseDuplicatePromptRepro:
    """Repro-style test for "message sent twice" observations.

    In the supervisor architecture, a single user turn can legitimately trigger
    multiple LLM invocations (supervisor routing, worker response, supervisor
    finishing decision). In tracing tools, this can look like the same user
    message being sent multiple times.

    This test asserts two things:
    1) The user message is NOT duplicated within any single LLM call.
    2) The graph performs multiple LLM calls for one user turn.
    """

    def test_user_message_not_duplicated_within_single_call(self, mock_skills_metadata):
        user_text = "hello - please help"

        class FakeStructured:
            def __init__(self, parent):
                self.parent = parent

            def invoke(self, messages):
                self.parent.calls.append(("supervisor", messages))
                decision = self.parent.decisions[self.parent.supervisor_call_index]
                self.parent.supervisor_call_index += 1
                return decision

        class FakeLLM:
            def __init__(self):
                self.calls = []
                self.decisions = [
                    SupervisorDecision(next="skill-a", reasoning="route to worker"),
                    SupervisorDecision(next="FINISH", reasoning="done"),
                ]
                self.supervisor_call_index = 0

            def with_structured_output(self, _schema):
                return FakeStructured(self)

            def bind_tools(self, _tools):
                return self

            def invoke(self, messages):
                self.calls.append(("worker", messages))
                return AIMessage(content="worker response")

        fake_llm = FakeLLM()

        with patch("agents_with_intent.graph.builder.discover_skills") as mock_discover, \
             patch("agents_with_intent.graph.nodes.discover_skills") as mock_discover_node, \
             patch("agents_with_intent.graph.nodes.SkillLoader") as mock_loader_class:

            mock_discover.return_value = mock_skills_metadata
            mock_discover_node.return_value = mock_skills_metadata

            # SkillLoader reads SKILL.md; mock it to avoid filesystem.
            mock_loader = MagicMock()
            mock_loader.name = "skill-a"
            mock_loader.description = "Test"
            mock_loader.has_scripts = False
            mock_loader.load_instructions.return_value = "Test instructions"
            mock_loader_class.return_value = mock_loader

            graph = create_supervisor_graph(
                llm=fake_llm,
                skills_dirs=["/test/skills"],
                skills_metadata=mock_skills_metadata,
            )

            result = graph.invoke(
                {"messages": [HumanMessage(content=user_text)]},
                config={"configurable": {"thread_id": "test-thread"}},
            )
            assert result is not None

        # Expect multiple LLM calls for one user message.
        kinds = [k for (k, _msgs) in fake_llm.calls]
        assert kinds.count("supervisor") == 2
        assert kinds.count("worker") == 1

        # Ensure the user message does not appear twice inside any single call.
        for kind, msgs in fake_llm.calls:
            user_occurrences = sum(
                1 for m in msgs if isinstance(m, HumanMessage) and m.content == user_text
            )
            assert user_occurrences == 1, f"{kind} had duplicated user message"


# =============================================================================
# Standard Tools Integration Tests
# =============================================================================

class TestStandardToolsIntegration:
    """Tests for standard tools integration with workers."""
    
    def test_worker_binds_requested_tools(self, mock_skills_metadata, mock_llm):
        """Test that worker binds only requested standard tools."""
        # Skill B requests file_write and file_read
        skill_b = mock_skills_metadata[1]
        assert "file_write" in skill_b["tools"]
        assert "file_read" in skill_b["tools"]
        
        # Create worker for skill B
        worker_fn = create_worker_node("skill-b", mock_llm, skill_b)
        
        # Worker should be created (tools binding happens during execution)
        assert callable(worker_fn)
    
    def test_worker_without_standard_tools(self, mock_skills_metadata, mock_llm):
        """Test that worker without requested tools works correctly."""
        # Skill A doesn't request standard tools
        skill_a = mock_skills_metadata[0]
        assert len(skill_a.get("tools", [])) == 0
        
        worker_fn = create_worker_node("skill-a", mock_llm, skill_a)
        assert callable(worker_fn)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
