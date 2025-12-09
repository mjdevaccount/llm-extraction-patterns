# Migration Guide: Local Model Patterns

## Overview

The local model patterns have been migrated from `universal_agent_nexus_examples/shared/workflows` into the new repository structure.

## Organization

### Structure

```
src/patterns/local_models/
├── README.md                    # Overview of local model patterns
├── WORKFLOW_README.md          # Original workflow documentation
├── REFACTORING_SUMMARY.md      # SOLID refactoring summary
├── SOLID_REFACTORING.md        # SOLID refactoring plan
│
├── abstractions.py             # SOLID abstractions (ILLMProvider, etc.)
├── strategies.py               # JSON repair and validation strategies
├── llm_adapter.py             # LangChain LLM adapter
│
├── nodes/
│   ├── base_nodes.py          # BaseNode interface
│   └── common_nodes.py        # IntelligenceNode, ExtractionNode, ValidationNode
│
├── workflows/
│   ├── workflow.py            # Workflow orchestration
│   ├── workflow_components.py # Workflow building components
│   └── helpers.py             # Helper workflows (ToolCalling, Conditional, etc.)
│
└── examples/
    └── example_07_refactored.py # Example usage
```

## Key Files

### Abstractions (`abstractions.py`)
- `ILLMProvider` - Abstract LLM interface (Dependency Inversion)
- `IJSONRepairStrategy` - JSON repair strategy interface
- `IValidationStrategy` - Validation strategy interface

### Strategies (`strategies.py`)
- `IncrementalRepairStrategy` - Mechanical JSON repair
- `LLMRepairStrategy` - LLM-based repair
- `RegexRepairStrategy` - Regex fallback
- `StrictValidationStrategy` - Fail fast
- `RetryValidationStrategy` - Retry with LLM repair
- `BestEffortValidationStrategy` - Best effort repairs

### Nodes (`nodes/`)
- `BaseNode` - Base interface for all nodes
- `IntelligenceNode` - Free-form reasoning (temp 0.7-0.8)
- `ExtractionNode` - Structured extraction (temp 0.1)
- `ValidationNode` - Validation with repair (temp 0.0)

### Workflows (`workflows/`)
- `Workflow` - Main workflow orchestrator
- `ToolCallingWorkflow` - Tool-calling loop pattern
- `ConditionalWorkflow` - Conditional branching pattern
- `SimpleQAWorkflow` - Simple Q&A pattern

## Usage

### Basic Usage

```python
from patterns.local_models import (
    IntelligenceNode,
    ExtractionNode,
    ValidationNode,
    LangChainLLMAdapter,
)
from patterns.local_models.workflows import Workflow

# Wrap LLM in adapter
llm_adapter = LangChainLLMAdapter(llm)

# Create nodes
intelligence = IntelligenceNode(
    llm=llm_adapter,
    prompt_template="Analyze: {input}",
    required_state_keys=["input"],
)

extraction = ExtractionNode(
    llm=llm_adapter,
    prompt_template="Extract: {analysis}",
    output_schema=MySchema,
    json_repair_strategies=["incremental_repair", "llm_repair"],
)

validation = ValidationNode(
    output_schema=MySchema,
    mode=ValidationMode.RETRY,  # Use retry strategy
)

# Create workflow
workflow = Workflow(
    name="analysis",
    nodes=[intelligence, extraction, validation],
    edges=[
        ("intelligence", "extraction"),
        ("extraction", "validation"),
    ]
)

# Execute
result = await workflow.invoke({"input": "..."})
```

## Differences from Simple Patterns

| Feature | Simple (`patterns/iev/`) | Advanced (`local_models/`) |
|---------|-------------------------|---------------------------|
| JSON Repair | Basic | Multiple strategies |
| Validation | Basic rules | Advanced with retry |
| Workflow | Simple graph | Full orchestration |
| Error Recovery | Basic | Sophisticated |
| SOLID Principles | Basic | Full SOLID compliance |
| Use Case | Cloud models | Local models |

## Import Path Changes

**Old:**
```python
from shared.workflows import IntelligenceNode, ExtractionNode
```

**New:**
```python
from patterns.local_models import IntelligenceNode, ExtractionNode
# or
from patterns.local_models.nodes.common_nodes import IntelligenceNode
```

## Notes

- All files have been copied as-is
- Import paths will need to be updated in examples
- The code is kept available for testing and design work
- See `LOCAL_MODELS.md` for detailed usage guide

