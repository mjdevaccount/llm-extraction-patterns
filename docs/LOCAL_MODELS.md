# Local Model Patterns Guide

## Overview

The `local_models/` directory contains advanced patterns specifically designed for **local models** (Ollama, Qwen, Mistral, etc.) that require more sophisticated error handling and hand-holding.

## When to Use Local Model Patterns

Use these patterns when:
- ✅ Working with local models (Ollama, etc.)
- ✅ Need advanced JSON repair strategies
- ✅ Require sophisticated validation with retry logic
- ✅ Want workflow orchestration with error recovery
- ✅ Building tool-calling or conditional workflows

**For cloud models (OpenAI, Anthropic)**, the simpler patterns in `patterns/iev/` are usually sufficient.

## Key Features

### 1. Advanced JSON Repair

Local models often produce malformed JSON. The local model patterns include multiple repair strategies:

```python
from patterns.local_models import (
    IncrementalRepairStrategy,
    LLMRepairStrategy,
    RegexRepairStrategy,
)

# Mechanical repair (fast)
incremental = IncrementalRepairStrategy()

# LLM-based repair (more reliable)
llm_repair = LLMRepairStrategy(llm_adapter)

# Regex fallback (last resort)
regex = RegexRepairStrategy()
```

### 2. Advanced Validation Strategies

```python
from patterns.local_models import (
    StrictValidationStrategy,
    RetryValidationStrategy,
    BestEffortValidationStrategy,
)

# Fail fast
strict = StrictValidationStrategy()

# Retry with LLM repair
retry = RetryValidationStrategy(llm_adapter, max_retries=3)

# Best effort (for non-critical pipelines)
best_effort = BestEffortValidationStrategy()
```

### 3. Workflow Orchestration

Advanced workflow system with error recovery:

```python
from patterns.local_models.workflows import Workflow
from patterns.local_models.nodes.common_nodes import (
    IntelligenceNode,
    ExtractionNode,
    ValidationNode,
)

workflow = Workflow(
    name="analysis-pipeline",
    nodes=[intelligence, extraction, validation],
    edges=[
        ("intelligence", "extraction"),
        ("extraction", "validation"),
    ]
)

result = await workflow.invoke(initial_state)
```

### 4. Helper Workflows

Pre-built patterns for common use cases:

```python
from patterns.local_models.workflows.helpers import (
    ToolCallingWorkflow,
    ConditionalWorkflow,
    SimpleQAWorkflow,
)

# Tool-calling loop
tool_workflow = ToolCallingWorkflow(
    name="research",
    llm=llm,
    tools=[search_tool, summarize_tool],
)

# Conditional branching
cond_workflow = ConditionalWorkflow(
    name="router",
    decision_node=classifier,
    branches={
        "urgent": [urgent_node],
        "normal": [normal_node],
    }
)
```

## Architecture

The local model patterns follow SOLID principles:

- **Single Responsibility**: Each strategy/node does one thing
- **Open/Closed**: Add new strategies without modifying existing code
- **Liskov Substitution**: Strategies are interchangeable
- **Interface Segregation**: Minimal required interfaces
- **Dependency Inversion**: Depend on abstractions, not implementations

## Migration Notes

This code was migrated from `universal_agent_nexus_examples/shared/workflows` and has been:
- Organized for better discoverability
- Documented for local model use cases
- Kept available for testing and design work

## Examples

See `src/patterns/local_models/examples/` for working examples.

## Comparison: Simple vs. Advanced

| Feature | Simple (`patterns/iev/`) | Advanced (`local_models/`) |
|---------|-------------------------|---------------------------|
| JSON Repair | Basic repair | Multiple strategies |
| Validation | Basic rules | Advanced strategies with retry |
| Workflow | Simple graph | Full orchestration |
| Error Recovery | Basic | Sophisticated |
| Use Case | Cloud models | Local models |

Choose based on your needs!

