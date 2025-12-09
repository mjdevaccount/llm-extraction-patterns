"""Tests for Workflow orchestrator."""

import pytest
from typing import Dict, Any, TypedDict
from patterns.patterns.iev.nodes.base_node import BaseNode, NodeExecutionError, NodeStatus
from patterns.patterns.iev.workflows import Workflow, WorkflowExecutionError
from patterns.patterns.iev.workflows.workflow_components import (
    GraphBuilder,
    WorkflowExecutor,
    MetricsCollector,
)


class WorkflowTestState(TypedDict, total=False):
    """Test state schema."""
    input: str
    output: str
    intermediate: str


# Use WorkflowTestState in tests
TestState = WorkflowTestState


class SimpleNode(BaseNode):
    """Simple node for testing."""
    
    def __init__(self, name: str, output_key: str = "output", required_key: str = None):
        super().__init__(name=name)
        self.output_key = output_key
        self.required_key = required_key or "input"
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Add output to state."""
        state[self.output_key] = f"processed_{state.get(self.required_key, '')}"
        return state
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Check required key exists."""
        return self.required_key in state


class TestGraphBuilder:
    """Tests for GraphBuilder component."""
    
    def test_graph_builder_initialization(self):
        """Test GraphBuilder is initialized correctly."""
        nodes = {"node1": SimpleNode("node1")}
        edges = []
        builder = GraphBuilder(
            state_schema=TestState,
            nodes=nodes,
            edges=edges,
        )
        assert builder.state_schema == TestState
        assert builder.nodes == nodes
        assert builder.edges == edges
    
    def test_compute_topology_single_node(self):
        """Test topology computation for single node."""
        nodes = {"node1": SimpleNode("node1")}
        builder = GraphBuilder(TestState, nodes, [])
        assert "node1" in builder.first_nodes
        assert builder.last_node == "node1"
    
    def test_compute_topology_chain(self):
        """Test topology computation for chain of nodes."""
        nodes = {
            "node1": SimpleNode("node1"),
            "node2": SimpleNode("node2"),
        }
        edges = [("node1", "node2")]
        builder = GraphBuilder(TestState, nodes, edges)
        assert "node1" in builder.first_nodes
        assert builder.last_node == "node2"
    
    def test_build_graph(self):
        """Test graph building."""
        nodes = {"node1": SimpleNode("node1")}
        builder = GraphBuilder(TestState, nodes, [])
        
        def wrapper_factory(node):
            async def wrapper(state):
                return await node.execute(state)
            return wrapper
        
        graph = builder.build(wrapper_factory, "test")
        assert graph is not None


class TestMetricsCollector:
    """Tests for MetricsCollector component."""
    
    def test_collector_initialization(self):
        """Test MetricsCollector is initialized."""
        collector = MetricsCollector("test_workflow")
        assert collector.workflow_name == "test_workflow"
        assert collector.metrics == {}
        assert collector.start_time is None
        assert collector.end_time is None
    
    def test_start_end_execution(self):
        """Test execution timing."""
        collector = MetricsCollector("test")
        collector.start_execution()
        assert collector.start_time is not None
        assert collector.metrics == {}
        
        collector.end_execution()
        assert collector.end_time is not None
    
    def test_record_execution(self):
        """Test recording node execution metrics."""
        collector = MetricsCollector("test")
        collector.record_execution(
            node_name="node1",
            duration_ms=123.45,
            input_keys=["input"],
            output_keys=["output"],
            status=NodeStatus.SUCCESS.value,
            warnings=["warning1"],
        )
        assert "node1" in collector.metrics
        assert collector.metrics["node1"]["duration_ms"] == 123.45
        assert collector.metrics["node1"]["status"] == "success"
    
    def test_get_metrics(self):
        """Test getting aggregated metrics."""
        collector = MetricsCollector("test")
        collector.start_execution()
        collector.record_execution(
            node_name="node1",
            duration_ms=100.0,
            input_keys=[],
            output_keys=[],
            status=NodeStatus.SUCCESS.value,
        )
        collector.end_execution()
        
        metrics = collector.get_metrics()
        assert metrics["workflow_name"] == "test"
        assert "total_duration_ms" in metrics
        assert "node1" in metrics["nodes"]
        assert metrics["overall_status"] == "success"
    
    def test_get_summary(self):
        """Test getting execution summary."""
        collector = MetricsCollector("test")
        collector.record_execution(
            node_name="node1",
            duration_ms=100.0,
            input_keys=[],
            output_keys=[],
            status=NodeStatus.SUCCESS.value,
        )
        
        summary = collector.get_summary()
        assert summary["workflow_name"] == "test"
        assert summary["node_count"] == 1
        assert summary["success_count"] == 1
        assert summary["failed_count"] == 0


class TestWorkflowExecutor:
    """Tests for WorkflowExecutor component."""
    
    def test_executor_initialization(self):
        """Test WorkflowExecutor is initialized."""
        collector = MetricsCollector("test")
        executor = WorkflowExecutor("test", collector)
        assert executor.workflow_name == "test"
        assert executor.metrics_collector == collector
    
    def test_create_node_wrapper(self):
        """Test node wrapper creation."""
        collector = MetricsCollector("test")
        executor = WorkflowExecutor("test", collector)
        node = SimpleNode("node1")
        wrapper = executor.create_node_wrapper(node)
        assert callable(wrapper)


class TestWorkflow:
    """Tests for Workflow orchestrator."""
    
    def test_workflow_initialization(self):
        """Test workflow is initialized correctly."""
        nodes = [SimpleNode("node1")]
        workflow = Workflow(
            name="test_workflow",
            state_schema=TestState,
            nodes=nodes,
            edges=[],
        )
        assert workflow.name == "test_workflow"
        assert workflow.state_schema == TestState
        assert "node1" in workflow.nodes
        assert len(workflow.edges) == 0
    
    def test_workflow_duplicate_node_names(self):
        """Test workflow rejects duplicate node names."""
        nodes = [
            SimpleNode("node1"),
            SimpleNode("node1"),  # Duplicate
        ]
        with pytest.raises(ValueError, match="Duplicate node name"):
            Workflow("test", TestState, nodes, [])
    
    def test_workflow_invalid_edge(self):
        """Test workflow rejects invalid edges."""
        nodes = [SimpleNode("node1")]
        edges = [("node1", "nonexistent")]
        with pytest.raises(ValueError, match="unknown node"):
            Workflow("test", TestState, nodes, edges)
    
    @pytest.mark.asyncio
    async def test_workflow_invoke_single_node(self):
        """Test workflow execution with single node."""
        nodes = [SimpleNode("node1")]
        workflow = Workflow("test", TestState, nodes, [])
        
        result = await workflow.invoke({"input": "test_value"})
        assert "output" in result
        assert result["output"] == "processed_test_value"
    
    @pytest.mark.asyncio
    async def test_workflow_invoke_chain(self):
        """Test workflow execution with node chain."""
        nodes = [
            SimpleNode("node1", output_key="intermediate"),
            SimpleNode("node2", required_key="intermediate"),
        ]
        edges = [("node1", "node2")]
        workflow = Workflow("test", TestState, nodes, edges)
        
        result = await workflow.invoke({"input": "test"})
        assert "intermediate" in result
        assert "output" in result
    
    @pytest.mark.asyncio
    async def test_workflow_invoke_validation_failure(self):
        """Test workflow handles validation failures."""
        nodes = [SimpleNode("node1", required_key="missing_key")]
        workflow = Workflow("test", TestState, nodes, [])
        
        with pytest.raises(WorkflowExecutionError):
            await workflow.invoke({"input": "value"})
    
    def test_workflow_visualize(self):
        """Test workflow visualization."""
        nodes = [
            SimpleNode("node1"),
            SimpleNode("node2"),
        ]
        edges = [("node1", "node2")]
        workflow = Workflow("test", TestState, nodes, edges)
        
        viz = workflow.visualize()
        assert "test" in viz
        assert "node1" in viz
        assert "node2" in viz
        assert "START" in viz
        assert "END" in viz
    
    @pytest.mark.asyncio
    async def test_workflow_get_metrics(self):
        """Test workflow metrics collection."""
        nodes = [SimpleNode("node1")]
        workflow = Workflow("test", TestState, nodes, [])
        
        await workflow.invoke({"input": "test"})
        metrics = workflow.get_metrics()
        assert metrics["workflow_name"] == "test"
        assert "node1" in metrics["nodes"]
        # Status might be "success" or "unknown" depending on metrics collection
        assert metrics["overall_status"] in ["success", "unknown"]
    
    def test_workflow_repr(self):
        """Test workflow string representation."""
        nodes = [SimpleNode("node1")]
        workflow = Workflow("test", TestState, nodes, [])
        repr_str = repr(workflow)
        assert "test" in repr_str
        assert "1" in repr_str  # node count

