"""
Workflow Components - Separated Concerns (SRP)

This module separates Workflow responsibilities into focused components:
  - GraphBuilder: Builds LangGraph from nodes and edges
  - WorkflowExecutor: Executes workflow and handles node wrappers
  - MetricsCollector: Collects and reports execution metrics

Single Responsibility Principle: Each component has one clear purpose.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypedDict, Tuple, Callable
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from ..nodes.base_nodes import (
    BaseNode,
    NodeExecutionError,
    NodeStatus,
    NodeMetrics,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Graph Builder (SRP: Builds graphs only)
# ============================================================================

class GraphBuilder:
    """
    Single Responsibility: Build LangGraph from nodes and edges.
    
    This class is responsible ONLY for graph construction.
    It does not handle execution, metrics, or error handling.
    """
    
    def __init__(
        self,
        state_schema: Type[TypedDict],
        nodes: Dict[str, BaseNode],
        edges: List[Tuple[str, str]],
    ):
        """
        Initialize graph builder.
        
        Args:
            state_schema: TypedDict defining state shape
            nodes: Dict of node_name -> BaseNode
            edges: List of (from_node, to_node) tuples
        """
        self.state_schema = state_schema
        self.nodes = nodes
        self.edges = edges
        self.first_nodes = []
        self.last_node = None
        
        # Compute topology
        self._compute_topology()
    
    def _compute_topology(self) -> None:
        """Compute workflow topology (first/last nodes)."""
        # Build adjacency list
        incoming = {name: set() for name in self.nodes.keys()}
        for from_node, to_node in self.edges:
            incoming[to_node].add(from_node)
        
        # Find first nodes (no incoming edges)
        self.first_nodes = [name for name, deps in incoming.items() if not deps]
        
        # Find last nodes (from end of edges list)
        if self.edges:
            self.last_node = self.edges[-1][1]
        else:
            self.last_node = self.first_nodes[0] if self.first_nodes else None
    
    def build(
        self,
        node_wrapper_factory: Callable[[BaseNode], Callable],
        workflow_name: str = "workflow",
    ) -> StateGraph:
        """
        Build and compile LangGraph.
        
        Args:
            node_wrapper_factory: Function that creates wrapper for a node
                Signature: (node: BaseNode) -> callable
            workflow_name: Name for logging
        
        Returns:
            Compiled LangGraph workflow
        """
        logger.info(
            f"[{workflow_name}] Building LangGraph with {len(self.nodes)} nodes"
        )
        
        graph = StateGraph(self.state_schema)
        
        # Add nodes
        for node_name, node in self.nodes.items():
            graph.add_node(
                node_name,
                node_wrapper_factory(node)
            )
        
        # Add edges
        # START -> first nodes
        for first_node in self.first_nodes:
            graph.add_edge(START, first_node)
        
        # Interior edges
        for from_node, to_node in self.edges:
            graph.add_edge(from_node, to_node)
        
        # Last node -> END
        if self.last_node:
            graph.add_edge(self.last_node, END)
        
        logger.info(f"[{workflow_name}] Graph compiled successfully")
        return graph.compile()


# ============================================================================
# Workflow Executor (SRP: Executes workflows only)
# ============================================================================

class WorkflowExecutor:
    """
    Single Responsibility: Execute workflow and handle node wrappers.
    
    This class is responsible ONLY for execution logic.
    It does not build graphs or collect metrics (delegates to MetricsCollector).
    """
    
    def __init__(
        self,
        workflow_name: str,
        metrics_collector: 'MetricsCollector',
    ):
        """
        Initialize executor.
        
        Args:
            workflow_name: Workflow identifier
            metrics_collector: Metrics collector instance
        """
        self.workflow_name = workflow_name
        self.metrics_collector = metrics_collector
    
    def create_node_wrapper(self, node: BaseNode) -> Callable:
        """
        Create wrapper function for node execution.
        
        Handles:
        - Input validation
        - Error handling
        - Metrics recording (delegates to MetricsCollector)
        
        Args:
            node: BaseNode instance
        
        Returns:
            Async function that wraps node.execute()
        """
        async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"[{self.workflow_name}] Executing {node.name}")
            
            try:
                # Validate input
                if not node.validate_input(state):
                    raise NodeExecutionError(
                        node_name=node.name,
                        reason="Input validation failed",
                        state=state
                    )
                
                # Execute
                state = await node.execute(state)
                
                # Record metrics
                self.metrics_collector.record_execution(
                    node_name=node.name,
                    duration_ms=node.metrics.duration_ms,
                    input_keys=node.metrics.input_keys,
                    output_keys=node.metrics.output_keys,
                    status=node.metrics.status.value,
                    warnings=node.metrics.warnings,
                )
                
                logger.info(
                    f"[{self.workflow_name}] {node.name} succeeded "
                    f"({node.metrics.duration_ms:.1f}ms)"
                )
                
                return state
            
            except Exception as e:
                logger.error(
                    f"[{self.workflow_name}] {node.name} failed: {e}"
                )
                
                # Record error metrics
                self.metrics_collector.record_execution(
                    node_name=node.name,
                    duration_ms=node.metrics.duration_ms,
                    input_keys=node.metrics.input_keys,
                    output_keys=node.metrics.output_keys,
                    status=NodeStatus.FAILED.value,
                    error=str(e),
                    warnings=node.metrics.warnings,
                )
                
                # Re-raise for Workflow to handle
                raise
        
        return wrapper
    
    async def execute(
        self,
        graph: StateGraph,
        initial_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute workflow graph.
        
        Args:
            graph: Compiled LangGraph
            initial_state: Starting state
        
        Returns:
            Final state after execution
        """
        logger.info(
            f"[{self.workflow_name}] Starting workflow execution"
        )
        
        final_state = await graph.ainvoke(initial_state)
        
        logger.info(
            f"[{self.workflow_name}] Workflow completed"
        )
        
        return final_state


# ============================================================================
# Metrics Collector (SRP: Collects metrics only)
# ============================================================================

class MetricsCollector:
    """
    Single Responsibility: Collect and report execution metrics.
    
    This class is responsible ONLY for metrics management.
    It does not execute workflows or build graphs.
    """
    
    def __init__(self, workflow_name: str):
        """
        Initialize metrics collector.
        
        Args:
            workflow_name: Workflow identifier
        """
        self.workflow_name = workflow_name
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def start_execution(self) -> None:
        """Mark start of workflow execution."""
        self.start_time = datetime.now()
        self.metrics = {}
    
    def end_execution(self) -> None:
        """Mark end of workflow execution."""
        self.end_time = datetime.now()
    
    def record_execution(
        self,
        node_name: str,
        duration_ms: float,
        input_keys: List[str],
        output_keys: List[str],
        status: str,
        warnings: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record metrics for a node execution.
        
        Args:
            node_name: Node identifier
            duration_ms: Execution duration in milliseconds
            input_keys: List of input state keys
            output_keys: List of output state keys
            status: Execution status (success, failed, etc.)
            warnings: Optional list of warnings
            error: Optional error message
        """
        self.metrics[node_name] = {
            "name": node_name,
            "status": status,
            "duration_ms": duration_ms,
            "input_keys": input_keys,
            "output_keys": output_keys,
            "warnings": warnings or [],
            "error_message": error,
        }
    
    def get_metrics(self, node_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for node(s).
        
        Args:
            node_name: Specific node, or None for all nodes
        
        Returns:
            Metrics dict
        """
        if node_name:
            return self.metrics.get(node_name, {})
        
        # Calculate total duration
        total_duration_ms = 0.0
        if self.start_time and self.end_time:
            total_duration_ms = (
                (self.end_time - self.start_time).total_seconds() * 1000
            )
        
        return {
            "workflow_name": self.workflow_name,
            "total_duration_ms": total_duration_ms,
            "node_count": len(self.metrics),
            "nodes": self.metrics,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get execution summary.
        
        Returns:
            Summary dict with counts and totals
        """
        total_duration = sum(
            m.get("duration_ms", 0.0) for m in self.metrics.values()
        )
        
        success_count = sum(
            1 for m in self.metrics.values()
            if m.get("status") == NodeStatus.SUCCESS.value
        )
        
        failed_count = sum(
            1 for m in self.metrics.values()
            if m.get("status") == NodeStatus.FAILED.value
        )
        
        return {
            "workflow_name": self.workflow_name,
            "total_duration_ms": total_duration,
            "node_count": len(self.metrics),
            "success_count": success_count,
            "failed_count": failed_count,
            "nodes": list(self.metrics.keys()),
        }

