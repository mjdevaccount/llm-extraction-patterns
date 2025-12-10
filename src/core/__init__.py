"""Core infrastructure shared across all patterns."""

from .llm_client import create_llm_client, LLMClient
from .llm_adapter import LangChainLLMAdapter
from .llm_helpers import invoke_llm, is_llm_client
from .llm_factory import get_reasoning_llm, get_extractor_llm, get_extractor_llm_from_formatter
from .extractor_formatters import (
    IExtractorFormatter,
    NuExtractFormatter,
    StandardExtractorFormatter,
)
from .mcp_tools import MCPToolkit, get_mcp_toolkit
from .memory import ConversationMemory, Message
from .types import NodeStatus, NodeMetrics, NodeExecutionError, BaseNode, NodeState
from .node_base import (
    # Status constants
    NodeResult,
    VerificationStatus,
    ValidationMode,
    # Utilities
    extract_json_from_response,
    extract_and_repair_json,
    parse_with_schema,
    check_state_error,
    require_state_keys,
    NodeTimer,
    format_validation_errors,
    get_schema_defaults,
    # Base classes
    LLMNode,
    IntelligenceNodeBase,
    ExtractionNodeBase,
    ValidationNodeBase,
    VerificationNodeBase,  # Alias
)
from .metrics import MetricsCollector
from .graph_builder import (
    WorkflowBuilder,
    ClassBasedWorkflowBuilder,
    create_linear_workflow,
    wrap_node_with_metrics,
)

# JSON Repair utilities (simple, synchronous)
from .json_repair import (
    repair_json,
    repair_unquoted_keys,
    repair_single_quotes,
    repair_trailing_commas,
    extract_json_object,
)

# Abstractions
from .abstractions import (
    ILLMProvider,
    IJSONRepairStrategy,
    IValidationStrategy,
    IRetryStrategy,
)

# Retry strategies
from .retry_strategies import (
    PromptRefinementRetry,
    LLMAssistedRepairRetry,
    CompositeRetryStrategy,
)

# Extraction strategies (async, schema-aware) - lazy import to avoid circular deps
# Import directly: from patterns.core.extraction_strategies import ...

__all__ = [
    # LLM & Infrastructure
    "create_llm_client",
    "LLMClient",
    "LangChainLLMAdapter",
    "invoke_llm",
    "is_llm_client",
    "get_reasoning_llm",
    "get_extractor_llm",
    "get_extractor_llm_from_formatter",
    "MCPToolkit",
    "get_mcp_toolkit",
    "ConversationMemory",
    "Message",
    "NodeStatus",
    "NodeMetrics",
    "NodeExecutionError",
    "BaseNode",
    "NodeState",
    "MetricsCollector",
    # Workflow builders
    "WorkflowBuilder",
    "ClassBasedWorkflowBuilder",
    "create_linear_workflow",
    "wrap_node_with_metrics",
    # Node base classes
    "NodeResult",
    "VerificationStatus",
    "ValidationMode",
    "extract_json_from_response",
    "extract_and_repair_json",
    "parse_with_schema",
    "check_state_error",
    "require_state_keys",
    "NodeTimer",
    "format_validation_errors",
    "get_schema_defaults",
    "LLMNode",
    "IntelligenceNodeBase",
    "ExtractionNodeBase",
    "ValidationNodeBase",
    "VerificationNodeBase",
    # Abstractions
    "ILLMProvider",
    "IJSONRepairStrategy",
    "IValidationStrategy",
    "IRetryStrategy",
    # JSON Repair (simple utilities)
    "repair_json",
    "repair_unquoted_keys",
    "repair_single_quotes",
    "repair_trailing_commas",
    "extract_json_object",
    # Retry Strategies
    "IRetryStrategy",
    "PromptRefinementRetry",
    "LLMAssistedRepairRetry",
    "CompositeRetryStrategy",
    # Extraction Strategies (import from extraction_strategies module directly)
    # "IncrementalRepairStrategy",  # Use: from patterns.core.extraction_strategies import ...
    # "LLMRepairStrategy",
    # "RegexRepairStrategy",
    # "StrictValidationStrategy",
    # "RetryValidationStrategy",
    # "BestEffortValidationStrategy",
]
