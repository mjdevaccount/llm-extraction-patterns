"""aistack-patterns: Learn LLM design patterns through implementation."""

import sys
from pathlib import Path

# Add src to path so we can import core
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.llm_client import create_llm_client, LLMClient
from core.mcp_tools import MCPToolkit, get_mcp_toolkit
from core.memory import ConversationMemory, Message
from core.types import NodeStatus, NodeMetrics, NodeExecutionError, BaseNode

# IEV Pattern exports
from .iev import (
    create_iev_graph,
    IEVState,
    intelligence_node,
    extraction_node,
    verification_node,
)

# Core base classes (for building new patterns)
from core.node_base import (
    LLMNode,
    IntelligenceNodeBase,
    ExtractionNodeBase,
    ValidationNodeBase,
    ValidationMode,
    VerificationStatus,
)

from core.graph_builder import (
    WorkflowBuilder,
    create_linear_workflow,
)

__all__ = [
    # Infrastructure
    "create_llm_client",
    "LLMClient",
    "MCPToolkit",
    "get_mcp_toolkit",
    "ConversationMemory",
    "Message",
    "NodeStatus",
    "NodeMetrics",
    "NodeExecutionError",
    "BaseNode",
    # IEV Pattern
    "create_iev_graph",
    "IEVState",
    "intelligence_node",
    "extraction_node",
    "verification_node",
    # Base classes for new patterns
    "LLMNode",
    "IntelligenceNodeBase",
    "ExtractionNodeBase",
    "ValidationNodeBase",
    "ValidationMode",
    "VerificationStatus",
    # Workflow builder
    "WorkflowBuilder",
    "create_linear_workflow",
]
