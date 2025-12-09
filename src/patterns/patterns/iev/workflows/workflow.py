"""
Workflow Orchestrator - SOLID Design

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

import logging
from typing import Any, Dict, List, Optional, Type, TypedDict, Tuple
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from ..nodes.base_node import (
    BaseNode,
    NodeExecutionError,
    NodeStatus,
)

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
        
        # SOLID: Initialize separated components
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
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph from nodes and edges.
        
        This is called lazily on first invoke().
        
        Returns:
            Compiled LangGraph workflow
        """
        return self._graph_builder.build(
            node_wrapper_factory=self._executor.create_node_wrapper,
            workflow_name=self.name,
        )
    
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
        self._metrics_collector.start_execution()
        
        logger.info(
            f"[{self.name}] Starting workflow execution"
        )
        
        try:
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
        if self._graph_builder.last_node:
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
        return self._metrics_collector.get_metrics()
    
    def __repr__(self) -> str:
        return (
            f"Workflow(name={self.name!r}, "
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)})"
        )
    
    def __str__(self) -> str:
        return self.visualize()

