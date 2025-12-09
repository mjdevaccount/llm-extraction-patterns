"""
Base Node Abstraction

Defines the fundamental interface that all workflow nodes must implement.

SOLID Principles:
  - Single Responsibility: Each node does ONE thing
  - Liskov Substitution: All nodes are interchangeable
  - Interface Segregation: Minimal required methods
  - Dependency Inversion: Nodes depend on abstractions

Design Patterns:
  - Strategy Pattern: Different node implementations for different tasks
  - Template Method: execute() calls internal methods in order
  - Dependency Injection: Dependencies passed via constructor
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TypedDict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeState(TypedDict, total=False):
    """
    Base TypedDict for workflow state.
    
    Subclass this in your workflow to define required and optional keys.
    
    Example:
        class MyWorkflowState(NodeState):
            event: Dict[str, Any]
            analysis: str
            extracted: Dict[str, Any]
    """
    messages: List[Any]  # LangChain message history
    metadata: Dict[str, Any]  # Workflow metadata
    error: Optional[str]  # Last error, if any


class NodeExecutionError(Exception):
    """
    Raised when a node fails to execute.
    
    Attributes:
        node_name: Name of the node that failed
        reason: Detailed error message
        state: State at time of failure (for debugging)
    """
    
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
    """
    Execution metrics for a node.
    
    Helps identify bottlenecks and validate performance.
    """
    name: str
    status: NodeStatus = NodeStatus.PENDING
    duration_ms: float = 0.0
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }


class BaseNode(ABC):
    """
    Abstract base class for all workflow nodes.
    
    Every workflow node must inherit from this and implement:
    - execute(): The actual work
    - validate_input(): Check preconditions
    
    Optional overrides:
    - on_error(): Custom error handling
    - get_metrics(): Performance introspection
    
    Single Responsibility:
        Each node does ONE thing well.
        - IntelligenceNode: Reasoning
        - ExtractionNode: Structured output
        - ValidationNode: Semantic checks
    
    Dependency Inversion:
        Nodes don't create their dependencies (LLMs, etc).
        Dependencies are injected via __init__.
    
    Example:
        class CustomNode(BaseNode):
            def __init__(self, llm, schema):
                super().__init__(name="custom")
                self.llm = llm
                self.schema = schema
            
            async def execute(self, state):
                response = await self.llm.ainvoke(...)
                state["output"] = response.content
                return state
            
            def validate_input(self, state):
                return "input_key" in state
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize node.
        
        Args:
            name: Unique identifier for this node in the workflow
            description: Human-readable description of what this node does
        """
        self.name = name
        self.description = description or f"{name} node"
        self.metrics = NodeMetrics(name=name)
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this node's work.
        
        This is the core method that must be implemented by subclasses.
        
        Args:
            state: Current workflow state (must contain all keys required by validate_input)
        
        Returns:
            Updated state dict with any new keys/modifications
        
        Raises:
            NodeExecutionError: If execution fails
        
        Design Pattern: Template Method
            The Workflow class calls this method and handles:
            - Input validation (validate_input() called first)
            - Metrics collection
            - Error handling (on_error() called on failure)
            
            Subclasses only need to implement execute().
        """
        pass
    
    @abstractmethod
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        Validate that state has all required keys for this node.
        
        Called by Workflow BEFORE execute().
        
        Args:
            state: Current workflow state
        
        Returns:
            True if state is valid, False otherwise
        
        Example:
            def validate_input(self, state):
                required = ["event", "analysis"]
                return all(k in state for k in required)
        
        Note:
            If this returns False, NodeExecutionError is raised with
            missing keys listed.
        """
        pass
    
    async def on_error(
        self,
        error: Exception,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle errors during execution.
        
        Called by Workflow if execute() raises an exception.
        Override to implement custom error handling (retry, repair, etc).
        
        Args:
            error: The exception that was raised
            state: State at time of error
        
        Returns:
            Updated state, or raise to propagate error
        
        Default behavior:
            Logs error and re-raises. Subclasses can override to:
            - Repair state
            - Retry with different parameters
            - Provide fallback values
        
        Example (in ValidationNode):
            async def on_error(self, error, state):
                # Try to repair broken JSON
                if "extracted" in state:
                    state["extracted"] = self._repair_json(state["extracted"])
                    return state
                raise error  # Can't recover, propagate
        """
        logger.error(
            f"[{self.name}] Error: {error}\n"
            f"State keys: {list(state.keys())}"
        )
        raise error
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics for this node.
        
        Used by Workflow for observability.
        
        Returns:
            Dict with execution stats (duration, status, warnings, etc)
        
        Example:
            {
                "name": "extraction",
                "status": "success",
                "duration_ms": 2345.5,
                "input_keys": ["analysis"],
                "output_keys": ["extracted"],
                "warnings": ["JSON required repair"],
            }
        """
        return self.metrics.to_dict()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
