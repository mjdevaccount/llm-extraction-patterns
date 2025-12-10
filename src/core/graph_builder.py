"""
LangGraph Workflow Builder

Reusable pattern for building state machine workflows with:
- Node wrapping with timing/metrics
- Fallback executor for non-LangGraph environments
- Metrics collection
- Error handling

New patterns just define their state and nodes, then use the builder.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None
    END = None
    HAS_LANGGRAPH = False

from .metrics import MetricsCollector

logger = logging.getLogger(__name__)

# Type for state dictionaries
State = TypeVar("State", bound=Dict[str, Any])

# Type for node functions: async (state, **params) -> dict
NodeFunc = Callable[..., Dict[str, Any]]


def wrap_node_with_metrics(
    node_func: NodeFunc,
    node_name: str,
    metrics: MetricsCollector,
    params: Dict[str, Any],
) -> Callable[[State], Dict[str, Any]]:
    """
    Wrap a node function with timing and metrics collection.
    
    Args:
        node_func: The node function to wrap
        node_name: Name for logging/metrics
        metrics: MetricsCollector instance
        params: Parameters to pass to node_func
    
    Returns:
        Wrapped async function for LangGraph
    """
    async def wrapped(state: State) -> Dict[str, Any]:
        start_time = time.time()
        try:
            result = await node_func(state, **params)
            duration = (time.time() - start_time) * 1000
            status = "success" if "error" not in result else "error"
            details = {}
            if "error" in result:
                details["error"] = result["error"]
            metrics.record(node_name, duration, status, details)
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            metrics.record(node_name, duration, "error", {"error": str(e)})
            logger.exception(f"[{node_name}] Error: {e}")
            return {"error": str(e)}
    
    return wrapped


class WorkflowBuilder:
    """
    Builder for LangGraph workflows.
    
    Handles:
    - Node registration with metrics wrapping
    - Edge definition
    - Fallback executor for non-LangGraph environments
    - Metrics collection
    
    Example:
        builder = WorkflowBuilder(MyState)
        builder.add_node("step1", step1_func, {"llm": llm})
        builder.add_node("step2", step2_func, {"schema": schema})
        builder.add_edge("step1", "step2")
        builder.add_edge("step2", END)
        builder.set_entry_point("step1")
        
        workflow = builder.build()
        result = await workflow({"input": "..."})
    """
    
    def __init__(self, state_type: type = None):
        """
        Initialize workflow builder.
        
        Args:
            state_type: TypedDict class for state (optional, for type hints)
        """
        self.state_type = state_type or dict
        self.metrics = MetricsCollector()
        self.nodes: Dict[str, Tuple[NodeFunc, Dict[str, Any]]] = {}
        self.edges: List[Tuple[str, str]] = []
        self.entry_point: Optional[str] = None
    
    def add_node(
        self,
        name: str,
        func: NodeFunc,
        params: Optional[Dict[str, Any]] = None,
    ) -> "WorkflowBuilder":
        """
        Add a node to the workflow.
        
        Args:
            name: Node identifier
            func: Async function (state, **params) -> dict
            params: Parameters to pass to the function
        
        Returns:
            self for chaining
        """
        self.nodes[name] = (func, params or {})
        return self
    
    def add_edge(self, from_node: str, to_node: str) -> "WorkflowBuilder":
        """
        Add an edge between nodes.
        
        Args:
            from_node: Source node name
            to_node: Target node name (or END constant)
        
        Returns:
            self for chaining
        """
        self.edges.append((from_node, to_node))
        return self
    
    def set_entry_point(self, node_name: str) -> "WorkflowBuilder":
        """
        Set the entry point node.
        
        Args:
            node_name: Name of the starting node
        
        Returns:
            self for chaining
        """
        self.entry_point = node_name
        return self
    
    def build(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Build the workflow.
        
        Returns:
            Async function that executes the workflow
        """
        if not self.entry_point:
            raise ValueError("Entry point not set. Call set_entry_point() first.")
        
        if not HAS_LANGGRAPH:
            logger.warning("LangGraph not installed. Using basic executor.")
            logger.warning("Install with: pip install langgraph")
            return self._build_simple_executor()
        
        return self._build_langgraph()
    
    def _build_simple_executor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Build a simple sequential executor without LangGraph."""
        # Determine execution order from edges
        order = self._get_execution_order()
        
        async def simple_executor(initial_state: Dict[str, Any]) -> Dict[str, Any]:
            state = dict(initial_state)
            
            for node_name in order:
                if node_name not in self.nodes:
                    continue
                
                func, params = self.nodes[node_name]
                wrapped = wrap_node_with_metrics(
                    func, node_name, self.metrics, params
                )
                result = await wrapped(state)
                state.update(result)
                
                # Stop on error
                if state.get("error"):
                    break
            
            state["metrics"] = self.metrics.get_summary()
            return state
        
        return simple_executor
    
    def _build_langgraph(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Build a LangGraph state machine."""
        graph = StateGraph(self.state_type)
        
        # Add wrapped nodes
        for node_name, (func, params) in self.nodes.items():
            wrapped = wrap_node_with_metrics(
                func, node_name, self.metrics, params
            )
            graph.add_node(node_name, wrapped)
        
        # Set entry point
        graph.set_entry_point(self.entry_point)
        
        # Add edges
        for from_node, to_node in self.edges:
            if to_node == "END" or to_node is END:
                graph.add_edge(from_node, END)
            else:
                graph.add_edge(from_node, to_node)
        
        compiled = graph.compile()
        
        async def invoke_with_metrics(initial_state: Dict[str, Any]) -> Dict[str, Any]:
            result = await compiled.ainvoke(initial_state)
            result["metrics"] = self.metrics.get_summary()
            return result
        
        return invoke_with_metrics
    
    def _get_execution_order(self) -> List[str]:
        """Get node execution order from edges (simple topological sort)."""
        if not self.entry_point:
            return list(self.nodes.keys())
        
        order = [self.entry_point]
        visited = {self.entry_point}
        
        # Follow edges
        current = self.entry_point
        for _ in range(len(self.nodes)):
            for from_node, to_node in self.edges:
                if from_node == current and to_node not in visited:
                    if to_node != "END" and to_node is not END:
                        order.append(to_node)
                        visited.add(to_node)
                        current = to_node
                        break
        
        return order


def create_linear_workflow(
    state_type: type,
    nodes: List[Tuple[str, NodeFunc, Dict[str, Any]]],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience function for creating a simple linear workflow.
    
    Args:
        state_type: TypedDict for state
        nodes: List of (name, func, params) tuples in execution order
    
    Returns:
        Async workflow function
    
    Example:
        workflow = create_linear_workflow(
            MyState,
            [
                ("step1", step1_func, {"llm": llm}),
                ("step2", step2_func, {"schema": schema}),
                ("step3", step3_func, {}),
            ]
        )
        result = await workflow({"input": "..."})
    """
    builder = WorkflowBuilder(state_type)
    
    for i, (name, func, params) in enumerate(nodes):
        builder.add_node(name, func, params)
        
        if i == 0:
            builder.set_entry_point(name)
        else:
            prev_name = nodes[i - 1][0]
            builder.add_edge(prev_name, name)
    
    # Add final edge to END
    if nodes:
        builder.add_edge(nodes[-1][0], "END")
    
    return builder.build()


# =============================================================================
# CLASS-BASED WORKFLOW BUILDER - For BaseNode instances
# =============================================================================

class ClassBasedWorkflowBuilder:
    """
    Builder for workflows using BaseNode class instances.
    
    Unlike WorkflowBuilder which takes functions, this takes BaseNode instances
    and calls their execute() method.
    
    Example:
        builder = ClassBasedWorkflowBuilder(MyState, "my_workflow")
        builder.add_node(IntelligenceNode(...))
        builder.add_node(ExtractionNode(...))
        builder.add_edge("intelligence", "extraction")
        builder.set_entry_point("intelligence")
        
        workflow = builder.build()
        result = await workflow({"input": "..."})
    """
    
    def __init__(self, state_type: type = None, name: str = "workflow"):
        """
        Initialize builder.
        
        Args:
            state_type: TypedDict class for state
            name: Workflow name for logging/metrics
        """
        self.state_type = state_type or dict
        self.name = name
        self.metrics = MetricsCollector(name)
        self.nodes: Dict[str, Any] = {}  # name -> BaseNode
        self.edges: List[Tuple[str, str]] = []
        self.entry_point: Optional[str] = None
    
    def add_node(self, node: Any) -> "ClassBasedWorkflowBuilder":
        """
        Add a BaseNode instance.
        
        Args:
            node: BaseNode instance (must have .name and .execute())
        
        Returns:
            self for chaining
        """
        self.nodes[node.name] = node
        return self
    
    def add_edge(self, from_node: str, to_node: str) -> "ClassBasedWorkflowBuilder":
        """Add edge between nodes."""
        self.edges.append((from_node, to_node))
        return self
    
    def set_entry_point(self, node_name: str) -> "ClassBasedWorkflowBuilder":
        """Set entry point node."""
        self.entry_point = node_name
        return self
    
    def _create_node_wrapper(self, node: Any) -> Callable:
        """Create wrapper for BaseNode.execute()."""
        async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Validate input
                if hasattr(node, 'validate_input') and not node.validate_input(state):
                    raise ValueError(f"Input validation failed for {node.name}")
                
                # Execute
                result = await node.execute(state)
                
                duration = (time.time() - start_time) * 1000
                
                # Record metrics
                self.metrics.record_node(
                    node_name=node.name,
                    duration_ms=duration,
                    status="success",
                    input_keys=getattr(node.metrics, 'input_keys', []) if hasattr(node, 'metrics') else [],
                    output_keys=getattr(node.metrics, 'output_keys', []) if hasattr(node, 'metrics') else [],
                    warnings=getattr(node.metrics, 'warnings', []) if hasattr(node, 'metrics') else [],
                )
                
                logger.info(f"[{self.name}] {node.name} succeeded ({duration:.1f}ms)")
                return result
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.metrics.record_node(
                    node_name=node.name,
                    duration_ms=duration,
                    status="failed",
                    error=str(e),
                )
                logger.error(f"[{self.name}] {node.name} failed: {e}")
                raise
        
        return wrapper
    
    def build(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Build the workflow."""
        if not self.entry_point:
            raise ValueError("Entry point not set")
        
        if not HAS_LANGGRAPH:
            return self._build_simple_executor()
        
        return self._build_langgraph()
    
    def _build_simple_executor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Build simple sequential executor."""
        order = self._get_execution_order()
        
        async def executor(initial_state: Dict[str, Any]) -> Dict[str, Any]:
            self.metrics.start()
            state = dict(initial_state)
            
            for node_name in order:
                if node_name not in self.nodes:
                    continue
                node = self.nodes[node_name]
                wrapper = self._create_node_wrapper(node)
                state = await wrapper(state)
                
                if state.get("error"):
                    break
            
            self.metrics.stop()
            state["metrics"] = self.metrics.get_workflow_metrics()
            return state
        
        return executor
    
    def _build_langgraph(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Build LangGraph workflow."""
        graph = StateGraph(self.state_type)
        
        for node_name, node in self.nodes.items():
            graph.add_node(node_name, self._create_node_wrapper(node))
        
        graph.set_entry_point(self.entry_point)
        
        for from_node, to_node in self.edges:
            if to_node == "END" or to_node is END:
                graph.add_edge(from_node, END)
            else:
                graph.add_edge(from_node, to_node)
        
        compiled = graph.compile()
        
        async def invoke_with_metrics(initial_state: Dict[str, Any]) -> Dict[str, Any]:
            self.metrics.start()
            result = await compiled.ainvoke(initial_state)
            self.metrics.stop()
            result["metrics"] = self.metrics.get_workflow_metrics()
            return result
        
        return invoke_with_metrics
    
    def _get_execution_order(self) -> List[str]:
        """Get node execution order from edges."""
        if not self.entry_point:
            return list(self.nodes.keys())
        
        order = [self.entry_point]
        visited = {self.entry_point}
        
        current = self.entry_point
        for _ in range(len(self.nodes)):
            for from_node, to_node in self.edges:
                if from_node == current and to_node not in visited:
                    if to_node != "END" and to_node is not END:
                        order.append(to_node)
                        visited.add(to_node)
                        current = to_node
                        break
        
        return order

