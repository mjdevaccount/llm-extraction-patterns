"""
Base Node Abstraction - Consolidated

Defines the fundamental interface that all workflow nodes must implement.
Works with both LLMClient (cloud) and LangChain ChatModel (local/brittle).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TypedDict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeState(TypedDict, total=False):
    """Base TypedDict for workflow state."""
    messages: List[Any]
    metadata: Dict[str, Any]
    error: Optional[str]


class NodeExecutionError(Exception):
    """Raised when a node fails to execute."""
    
    def __init__(self, node_name: str, reason: str, state: Dict[str, Any] = None):
        self.node_name = node_name
        self.reason = reason
        self.state = state or {}
        super().__init__(
            f"[{node_name}] {reason}\n"
            f"State keys: {list(state.keys()) if state else 'N/A'}"
        )


class NodeStatus(str, Enum):
    """Node execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeMetrics:
    """Execution metrics for a node."""
    name: str
    status: NodeStatus = NodeStatus.PENDING
    duration_ms: float = 0.0
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)  # Additional metrics
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "extra": self.extra,
        }


class BaseNode(ABC):
    """Abstract base class for all workflow nodes."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description or f"{name} node"
        self.metrics = NodeMetrics(name=name)
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this node's work."""
        pass
    
    @abstractmethod
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate that state has all required keys."""
        pass
    
    async def on_error(self, error: Exception, state: Dict[str, Any]) -> Exception:
        """Handle errors during execution."""
        logger.error(
            f"[{self.name}] Error: {error}\n"
            f"State keys: {list(state.keys())}"
        )
        return error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for this node."""
        return self.metrics.to_dict()

