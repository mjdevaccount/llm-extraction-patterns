"""
Workflow Pattern - Class-based workflow orchestration.

Compose BaseNode instances into LangGraph workflows with:
- Automatic graph building from nodes + edges
- Validation, error handling, metrics
- ASCII visualization

Uses core.ClassBasedWorkflowBuilder under the hood.
"""

from .workflow import Workflow, WorkflowExecutionError

# Re-export core utilities for convenience
from core.graph_builder import ClassBasedWorkflowBuilder
from core.metrics import MetricsCollector
from core.types import BaseNode, NodeExecutionError, NodeStatus, NodeMetrics

__all__ = [
    # Pattern
    "Workflow",
    "WorkflowExecutionError",
    # Core re-exports
    "ClassBasedWorkflowBuilder",
    "MetricsCollector",
    "BaseNode",
    "NodeExecutionError",
    "NodeStatus",
    "NodeMetrics",
]
