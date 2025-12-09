"""Shared type definitions for all patterns."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeStatus(str, Enum):
    """Status of a node execution."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class NodeMetrics:
    """Metrics collected during node execution."""

    status: NodeStatus
    duration_ms: float = 0.0
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class NodeExecutionError(Exception):
    """Error raised during node execution."""

    def __init__(self, node_name: str, reason: str, state: Dict[str, Any]):
        self.node_name = node_name
        self.reason = reason
        self.state = state
        super().__init__(f"[{node_name}] {reason}")

