"""
Workflow Orchestrator

Compose BaseNode instances into LangGraph workflows.

Design:
  - Takes list of nodes + edges
  - Builds LangGraph automatically
  - Handles validation, error handling, metrics collection
  - Provides observability (visualization, tracing)

SOLID Principles:
  - Single Responsibility: Workflow orchestrates, nodes execute
  - Open/Closed: Add nodes via configuration, not code changes
  - Liskov Substitution: Any BaseNode subclass works
  - Interface Segregation: Workflow has minimal interface
  - Dependency Inversion: Depends on BaseNode abstraction

Example:
    workflow = Workflow(
        name="adoption-analysis",
        state_schema=AgentState,
        nodes=[intelligence_node, extraction_node, validation_node],
        edges=[
            ("intelligence", "extraction"),
            ("extraction", "validation"),
        ]
    )
    
    result = await workflow.invoke(initial_state)
    print(workflow.visualize())
    print(workflow.get_metrics())
"""

import asyncio
import logging
from typing import (
    Any, Dict, List, Optional, Type, TypedDict, Tuple
)
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from ..nodes.base_nodes import (
    BaseNode,
    NodeExecutionError,
    NodeStatus,
)

# SOLID Refactoring: Import separated components
from .workflow_components import (
    GraphBuilder,
    WorkflowExecutor,
    MetricsCollector,
)

logger = logging.getLogger(__name__)


class WorkflowExecutionError(Exception):
    """
    Raised when workflow execution fails.
    
    Attributes:
        workflow_name: Name of the workflow
        failed_node: Name of node that failed
        reason: Error message
        state_at_failure: State snapshot
        metrics: Metrics at time of failure
    """
    
    def __init__(
        self,
        workflow_name: str,
        failed_node: str,
        reason: str,
        state_at_failure: Dict[str, Any] = None,
        metrics: Dict[str, Any] = None,
    ):
        self.workflow_name = workflow_name
        self.failed_node = failed_node
        self.reason = reason
        self.state_at_failure = state_at_failure or {}
        self.metrics = metrics or {}
        super().__init__(
            f"[{workflow_name}] {failed_node} failed: {reason}"
        )


class Workflow:
    """
    Orchestrates BaseNode instances into a LangGraph workflow.
    
    Responsibilities:
        1. Build LangGraph from nodes + edges
        2. Validate state at each node
        3. Handle errors and metrics
        4. Provide observability
    
    Usage:
        # Define nodes (dependency injection)
        intelligence = IntelligenceNode(llm=..., prompt_template=...)
        extraction = ExtractionNode(llm=..., output_schema=...)
        validation = ValidationNode(output_schema=...)
        
        # Compose into workflow
        workflow = Workflow(
            name="analysis",
            state_schema=MyState,
            nodes=[intelligence, extraction, validation],
            edges=[
                ("intelligence", "extraction"),
                ("extraction", "validation"),
            ]
        )
        
        # Run
        result = await workflow.invoke(initial_state)
        
        # Observe
        print(workflow.visualize())
        print(workflow.get_metrics())
    """
    
    def __init__(
        self,
        name: str,
        state_schema: Type[TypedDict],
        nodes: List[BaseNode],
        edges: List[Tuple[str, str]],
    ):
        """
        Initialize workflow.
        
        Args:
            name: Workflow identifier
            state_schema: TypedDict defining state shape
            nodes: List of BaseNode instances
            edges: List of (from_node, to_node) tuples
        
        Raises:
            ValueError: If edges reference non-existent nodes
        """
        self.name = name
        self.state_schema = state_schema
        self.edges = edges
        
        # Index nodes by name
        self.nodes: Dict[str, BaseNode] = {}
        for node in nodes:
            if node.name in self.nodes:
                raise ValueError(
                    f"Duplicate node name: {node.name}"
                )
            self.nodes[node.name] = node
        
        # Validate edges
        node_names = set(self.nodes.keys())
        for from_node, to_node in edges:
            if from_node not in node_names:
                raise ValueError(
                    f"Edge references unknown node: {from_node}. "
                    f"Available: {node_names}"
                )
            if to_node not in node_names:
                raise ValueError(
                    f"Edge references unknown node: {to_node}. "
                    f"Available: {node_names}"
                )
        
        # SOLID Refactoring: Initialize separated components
        self._graph_builder = GraphBuilder(
            state_schema=state_schema,
            nodes=self.nodes,
            edges=edges,
        )
        self._metrics_collector = MetricsCollector(workflow_name=name)
        self._executor = WorkflowExecutor(
            workflow_name=name,
            metrics_collector=self._metrics_collector,
        )
        
        # Build graph (lazy - only when needed)
        self._graph = None
        self._start_time = None
        self._metrics = {}
    
    def _compute_topology(self) -> None:
        """
        Compute workflow topology (first/last nodes, execution order).
        """
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
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph from nodes and edges.
        
        This is called lazily on first invoke().
        
        Returns:
            Compiled LangGraph workflow
        """
        # SOLID Refactoring: Use GraphBuilder
        return self._graph_builder.build(
            node_wrapper_factory=self._executor.create_node_wrapper,
            workflow_name=self.name,
        )
        logger.info(
            f"[{self.name}] Building LangGraph with {len(self.nodes)} nodes"
        )
        
        graph = StateGraph(self.state_schema)
        
        # Add nodes (wrap to handle async + metrics)
        for node_name, node in self.nodes.items():
            graph.add_node(
                node_name,
                self._make_node_wrapper(node)
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
        
        logger.info(f"[{self.name}] Graph compiled successfully")
        return graph.compile()
    
    def _make_node_wrapper(self, node: BaseNode) -> Any:
        """
        Wrap node execution to handle:
        - Input validation
        - Async/sync conversion
        - Error handling
        - Metrics collection
        """
        async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"[{self.name}] Executing {node.name}")
            
            try:
                # Validate input
                if not node.validate_input(state):
                    missing = [
                        k for k in state.keys() 
                        if k not in state
                    ]
                    raise NodeExecutionError(
                        node_name=node.name,
                        reason=f"Input validation failed",
                        state=state
                    )
                
                # Execute
                state = await node.execute(state)
                
                # Metrics collected by MetricsCollector
                
                logger.info(
                    f"[{self.name}] {node.name} succeeded "
                    f"({node.metrics.duration_ms:.1f}ms)"
                )
                
                return state
            
            except Exception as e:
                logger.error(
                    f"[{self.name}] {node.name} failed: {e}"
                )
                
                # Wrap in WorkflowExecutionError
                metrics = node.get_metrics()
                raise WorkflowExecutionError(
                    workflow_name=self.name,
                    failed_node=node.name,
                    reason=str(e),
                    state_at_failure=state,
                    metrics=metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics,
                )
        
        return wrapper
    
    @property
    def graph(self) -> StateGraph:
        """
        Lazy-compile graph on first access.
        """
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph
    
    async def invoke(
        self,
        initial_state: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute workflow.
        
        Args:
            initial_state: Starting state dict
            verbose: Enable debug logging
        
        Returns:
            Final state after all nodes execute
        
        Raises:
            WorkflowExecutionError: If any node fails
        
        Example:
            result = await workflow.invoke({
                "event": {"name": "AI breakthrough"},
                "scenario": "market impact"
            })
            print(result["analysis"])
            print(result["validated"])
        """
        # SOLID Refactoring: Use MetricsCollector if available
        if SOLID_COMPONENTS_AVAILABLE:
            self._metrics_collector.start_execution()
        else:
            self._start_time = datetime.now()
            self._metrics = {}
        
        logger.info(
            f"[{self.name}] Starting workflow execution"
        )
        
        try:
            # SOLID Refactoring: Use WorkflowExecutor
            final_state = await self._executor.execute(
                graph=self.graph,
                initial_state=initial_state,
            )
            self._metrics_collector.end_execution()
            
            elapsed = (
                datetime.now() - self._metrics_collector.start_time
            ).total_seconds() if self._metrics_collector.start_time else 0.0
            
            logger.info(
                f"[{self.name}] Workflow completed "
                f"({elapsed:.2f}s, {len(self.nodes)} nodes)"
            )
            
            return final_state
        
        except WorkflowExecutionError as e:
            logger.error(
                f"[{self.name}] Workflow failed at {e.failed_node}"
            )
            raise
        
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error: {e}")
            raise WorkflowExecutionError(
                workflow_name=self.name,
                failed_node="unknown",
                reason=f"Unexpected error: {e}",
                state_at_failure=initial_state,
            )
    
    def visualize(self) -> str:
        """
        Generate ASCII visualization of workflow.
        
        Returns:
            Multi-line string showing graph structure
        
        Example output:
            Workflow: adoption-analysis
            State: AgentState
            Nodes: 3 (intelligence, extraction, validation)
            Edges: 2
            
            Graph:
              START
                |
                v
            [intelligence]
                |
                v
            [extraction]
                |
                v
            [validation]
                |
                v
              END
        """
        lines = [
            f"Workflow: {self.name}",
            f"State: {self.state_schema.__name__}",
            f"Nodes: {len(self.nodes)} ({', '.join(self.nodes.keys())})",
            f"Edges: {len(self.edges)}",
            "",
            "Graph:",
            "  START",
        ]
        
        # Simple linear visualization
        for from_node, to_node in self.edges:
            lines.append("    |")
            lines.append("    v")
            lines.append(f"  [{from_node}]")
        
        # Last node
        if self.last_node:
            lines.append("    |")
            lines.append("    v")
            lines.append("  END")
        
        return "\n".join(lines)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated workflow metrics.
        
        Returns:
            Dict with per-node metrics and totals
        
        Example:
            {
                "workflow_name": "adoption-analysis",
                "total_duration_ms": 5234.2,
                "status": "success",
                "nodes": {
                    "intelligence": {
                        "status": "success",
                        "duration_ms": 1234.5,
                        ...
                    },
                    "extraction": {...},
                    "validation": {...}
                }
            }
        """
        # SOLID Refactoring: Use MetricsCollector
        return self._metrics_collector.get_metrics()
        total_duration = 0.0
        all_warnings = []
        
        node_metrics = {}
        for node_name, metrics in self._metrics.items():
            node_metrics[node_name] = metrics
            total_duration += metrics.get("duration_ms", 0)
            all_warnings.extend(metrics.get("warnings", []))
        
        # Determine overall status
        statuses = [m.get("status") for m in self._metrics.values()]
        if NodeStatus.FAILED.value in statuses:
            overall_status = "failed"
        elif NodeStatus.SUCCESS.value in statuses:
            overall_status = "success"
        else:
            overall_status = "unknown"
        
        return {
            "workflow_name": self.name,
            "overall_status": overall_status,
            "total_duration_ms": total_duration,
            "nodes_executed": len(self._metrics),
            "total_warnings": len(all_warnings),
            "nodes": node_metrics,
            "warnings": all_warnings,
        }
    
    def __repr__(self) -> str:
        return (
            f"Workflow(name={self.name!r}, "
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)})"
        )
    
    def __str__(self) -> str:
        return self.visualize()
