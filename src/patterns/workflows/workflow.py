"""
Workflow Orchestrator Pattern

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
from typing import Any, Dict, List, Type, TypedDict, Tuple

from core.types import BaseNode
from core.graph_builder import ClassBasedWorkflowBuilder
from core.metrics import MetricsCollector

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
                raise ValueError(f"Duplicate node name: {node.name}")
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
        
        # Use core builder
        self._builder = ClassBasedWorkflowBuilder(state_schema, name)
        
        # Add nodes
        for node in nodes:
            self._builder.add_node(node)
        
        # Compute entry point (nodes with no incoming edges)
        incoming = {n: set() for n in node_names}
        for from_node, to_node in edges:
            incoming[to_node].add(from_node)
        
        first_nodes = [n for n, deps in incoming.items() if not deps]
        if first_nodes:
            self._builder.set_entry_point(first_nodes[0])
        
        # Add edges
        for from_node, to_node in edges:
            self._builder.add_edge(from_node, to_node)
        
        # Find last node and connect to END
        if edges:
            last_node = edges[-1][1]
            self._builder.add_edge(last_node, "END")
        
        self._graph = None
        self._last_node = edges[-1][1] if edges else None
    
    @property
    def graph(self):
        """Lazy-compile graph on first access."""
        if self._graph is None:
            self._graph = self._builder.build()
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
        """
        logger.info(f"[{self.name}] Starting workflow execution")
        
        try:
            final_state = await self.graph(initial_state)
            
            total_ms = final_state.get("metrics", {}).get("total_duration_ms", 0)
            logger.info(
                f"[{self.name}] Workflow completed "
                f"({total_ms/1000:.2f}s, {len(self.nodes)} nodes)"
            )
            
            return final_state
        
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
        if self._last_node:
            lines.append("    |")
            lines.append("    v")
            lines.append("  END")
        
        return "\n".join(lines)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated workflow metrics.
        
        Returns:
            Dict with per-node metrics and totals
        """
        return self._builder.metrics.get_workflow_metrics()
    
    def __repr__(self) -> str:
        return (
            f"Workflow(name={self.name!r}, "
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)})"
        )
    
    def __str__(self) -> str:
        return self.visualize()
