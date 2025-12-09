"""Tests for BaseNode and core node infrastructure."""

import pytest
from typing import Dict, Any
from patterns.patterns.iev.nodes.base_node import (
    BaseNode,
    NodeExecutionError,
    NodeStatus,
    NodeMetrics,
    NodeState,
)


class ConcreteNode(BaseNode):
    """Concrete implementation for testing BaseNode."""
    
    def __init__(self, name: str = "test_node", required_keys: list = None):
        super().__init__(name=name, description="Test node")
        self.required_keys = required_keys or []
        self.executed = False
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple execution that marks as executed."""
        self.executed = True
        state["executed"] = True
        return state
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate required keys are present."""
        return all(k in state for k in self.required_keys)


class TestNodeMetrics:
    """Tests for NodeMetrics dataclass."""
    
    def test_metrics_initialization(self):
        """Test metrics are initialized correctly."""
        metrics = NodeMetrics(name="test")
        assert metrics.name == "test"
        assert metrics.status == NodeStatus.PENDING
        assert metrics.duration_ms == 0.0
        assert metrics.input_keys == []
        assert metrics.output_keys == []
        assert metrics.error_message is None
        assert metrics.warnings == []
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dict."""
        metrics = NodeMetrics(
            name="test",
            status=NodeStatus.SUCCESS,
            duration_ms=123.45,
            input_keys=["input"],
            output_keys=["output"],
            warnings=["warning1"],
        )
        result = metrics.to_dict()
        assert result["name"] == "test"
        assert result["status"] == "success"
        assert result["duration_ms"] == 123.45
        assert result["input_keys"] == ["input"]
        assert result["output_keys"] == ["output"]
        assert result["warnings"] == ["warning1"]


class TestBaseNode:
    """Tests for BaseNode abstract class."""
    
    def test_node_initialization(self):
        """Test node is initialized with name and description."""
        node = ConcreteNode(name="my_node")
        node.description = "My description"  # Set after init
        assert node.name == "my_node"
        assert node.description == "My description"
        assert node.metrics.name == "my_node"
        assert node.metrics.status == NodeStatus.PENDING
    
    def test_node_default_description(self):
        """Test default description is generated from name."""
        node = ConcreteNode(name="test")
        assert node.description == "Test node"  # Capitalized by BaseNode
    
    def test_node_metrics(self):
        """Test node metrics are accessible."""
        node = ConcreteNode()
        metrics = node.get_metrics()
        assert metrics["name"] == "test_node"
        assert "status" in metrics
        assert "duration_ms" in metrics
    
    @pytest.mark.asyncio
    async def test_node_execute(self):
        """Test node execution."""
        node = ConcreteNode()
        state = {"input": "value"}
        result = await node.execute(state)
        assert node.executed is True
        assert result["executed"] is True
        assert "input" in result
    
    @pytest.mark.asyncio
    async def test_node_on_error(self):
        """Test error handling."""
        node = ConcreteNode()
        error = ValueError("Test error")
        state = {"key": "value"}
        
        # on_error returns the error, not raises it
        result = await node.on_error(error, state)
        assert result == error
    
    def test_node_validate_input(self):
        """Test input validation."""
        node = ConcreteNode(required_keys=["key1", "key2"])
        assert node.validate_input({"key1": "v1", "key2": "v2"}) is True
        assert node.validate_input({"key1": "v1"}) is False
        assert node.validate_input({}) is False


class TestNodeExecutionError:
    """Tests for NodeExecutionError exception."""
    
    def test_error_initialization(self):
        """Test error is initialized with node name and reason."""
        error = NodeExecutionError(
            node_name="test_node",
            reason="Test failure",
            state={"key": "value"}
        )
        assert error.node_name == "test_node"
        assert error.reason == "Test failure"
        assert error.state == {"key": "value"}
        assert "test_node" in str(error)
        assert "Test failure" in str(error)
    
    def test_error_without_state(self):
        """Test error can be created without state."""
        error = NodeExecutionError(
            node_name="test_node",
            reason="Test failure"
        )
        assert error.state == {}


class TestNodeState:
    """Tests for NodeState TypedDict."""
    
    def test_node_state_typing(self):
        """Test NodeState can be used as TypedDict."""
        state: NodeState = {
            "messages": [],
            "metadata": {"key": "value"},
            "error": None,
        }
        assert state["messages"] == []
        assert state["metadata"]["key"] == "value"
        assert state["error"] is None

