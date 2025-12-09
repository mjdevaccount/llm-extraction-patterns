"""
Local Model Patterns

Advanced patterns and utilities for local models (Ollama, etc.) that require
more sophisticated error handling, JSON repair, and workflow orchestration.

These are optional - use when working with local models or when you need
advanced features like:
- Sophisticated JSON repair strategies
- Advanced validation with retry logic
- Workflow orchestration with error recovery
- Tool-calling and conditional workflows
"""

# Core abstractions
from .abstractions import (
    ILLMProvider,
    IJSONRepairStrategy,
    IValidationStrategy,
)

# Strategy implementations
from .strategies import (
    IncrementalRepairStrategy,
    LLMRepairStrategy,
    RegexRepairStrategy,
    StrictValidationStrategy,
    RetryValidationStrategy,
    BestEffortValidationStrategy,
)

# LLM adapter
from .llm_adapter import LangChainLLMAdapter

# Advanced nodes (for local models)
from .nodes import (
    IntelligenceNode,
    ExtractionNode,
    ValidationNode,
    ValidationMode,
)
from .nodes.base_nodes import (
    BaseNode,
    NodeExecutionError,
    NodeState,
    NodeStatus,
    NodeMetrics,
)

# Advanced workflows
from .workflows.workflow import Workflow, WorkflowExecutionError
from .workflows.helpers import (
    ToolCallingWorkflow,
    ConditionalWorkflow,
    SimpleQAWorkflow,
)

__all__ = [
    # Abstractions
    "ILLMProvider",
    "IJSONRepairStrategy",
    "IValidationStrategy",
    # Strategies
    "IncrementalRepairStrategy",
    "LLMRepairStrategy",
    "RegexRepairStrategy",
    "StrictValidationStrategy",
    "RetryValidationStrategy",
    "BestEffortValidationStrategy",
    # Adapter
    "LangChainLLMAdapter",
    # Nodes
    "BaseNode",
    "NodeExecutionError",
    "NodeState",
    "NodeStatus",
    "NodeMetrics",
    "IntelligenceNode",
    "ExtractionNode",
    "ValidationNode",
    "ValidationMode",
    # Workflows
    "Workflow",
    "WorkflowExecutionError",
    "ToolCallingWorkflow",
    "ConditionalWorkflow",
    "SimpleQAWorkflow",
]

