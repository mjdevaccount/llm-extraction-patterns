"""
Workflow Components - Using Core Utilities

This module provides workflow-specific wrappers around core utilities:
  - GraphBuilder: Uses core.ClassBasedWorkflowBuilder under the hood
  - WorkflowExecutor: Thin wrapper for execution
  - MetricsCollector: Re-exported from core.metrics

Single Responsibility Principle: Each component has one clear purpose.
"""

# Re-export from core - these are the canonical implementations
from core.metrics import MetricsCollector
from core.graph_builder import ClassBasedWorkflowBuilder
from core.types import BaseNode, NodeExecutionError, NodeStatus, NodeMetrics

# Alias for backwards compatibility
GraphBuilder = ClassBasedWorkflowBuilder

__all__ = [
    "MetricsCollector",
    "ClassBasedWorkflowBuilder",
    "GraphBuilder",  # alias
    "BaseNode",
    "NodeExecutionError",
    "NodeStatus",
    "NodeMetrics",
]
