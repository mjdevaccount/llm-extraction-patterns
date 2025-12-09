# Retry Strategies Guide

This guide explains how to use retry strategies in the IEV extraction phase to handle JSON parsing failures.

**Quick Summary:**
When LLM extraction produces invalid JSON, retry strategies automatically refine the prompt or ask the LLM to fix its output. This dramatically improves success rates for brittle LLMs.

---

## Table of Contents

1. [What is a Retry Strategy?](#what-is-a-retry-strategy)
2. [Built-in Strategies](#built-in-strategies)
3. [Using Strategies in IEV](#using-strategies-in-iev)
4. [Creating Custom Strategies](#creating-custom-strategies)
5. [Retry Flow Diagram](#retry-flow-diagram)
6. [Real-World Examples](#real-world-examples)
7. [Best Practices](#best-practices)

---

## What is a Retry Strategy?

A retry strategy defines what to do when extraction fails:

```
Attempt 1: Generate JSON
    │
    ├─ Success? → Return data
    └─ Failure? → Retry strategy

Retry 1: Refine prompt or repair
    │
    ├─ Success? → Return data
    └─ Failure? → Next retry

Retry 2: Try different approach
    │
    ├─ Success? → Return data
    └─ Failure? → Next retry

...

Giveup: Return error
```

**Why it matters:**
- Local LLMs (Mistral 7B, Llama 2) produce invalid JSON ~30-40% of the time
- Simple regex repairs fail ~50% of the time
- Retry with prompt refinement recovers another ~20-30%
- LLM-assisted repair recovers another ~10-15%
- **Total success: 95%+** instead of 60% with basic repair

---

## Built-in Strategies

### 1. PromptRefinementRetry

**Approach:** Refine extraction prompt with stricter instructions

**Flow:**
```
Attempt 1: "Extract data matching schema"
    │ Failure
    └─→ Attempt 2: "Extract ONLY valid JSON, no explanation"
           │ Failure
           └─→ Attempt 3: "Return ONLY raw JSON, no markdown"
                  │ Success or Failure
                  └─→ Stop
```

**Use when:** You want to fix the prompt without additional LLM calls

**Code:**
```python
from patterns.patterns.iev.retry_strategies import PromptRefinementRetry

strategy = PromptRefinementRetry(max_retries=3)

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    retry_strategy=strategy,
)
```

**Pros:**
- ✅ Fast (reuses first LLM response)
- ✅ Inexpensive (no extra API calls)
- ✅ Often works for local LLMs

**Cons:**
- ❌ Doesn't help if LLM fundamentally misunderstood request

---

### 2. LLMAssistedRepairRetry

**Approach:** Ask the LLM to repair its own broken JSON

**Flow:**
```
Attempt 1: "Extract data"
    │ Failure
    └─→ Repair 1: "Fix this broken JSON: {...}"
           │ Failure
           └─→ Repair 2: "Try again with schema: {...}"
                  │ Success or Failure
                  └─→ Stop
```

**Use when:** You want the LLM to fix itself

**Code:**
```python
from patterns.patterns.iev.retry_strategies import LLMAssistedRepairRetry

strategy = LLMAssistedRepairRetry(max_retries=2)

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    retry_strategy=strategy,
)
```

**Pros:**
- ✅ LLM understands what went wrong
- ✅ Works well for simple JSON
- ✅ High success rate for cloud LLMs

**Cons:**
- ❌ Makes extra API calls (more expensive)
- ❌ Slow (each retry is another LLM call)

---

### 3. CompositeRetryStrategy (Recommended Default)

**Approach:** Try prompt refinement first, then LLM repair

**Flow:**
```
Attempt 1: Original prompt
    │ Failure
    └─→ Refine 1: Stricter instructions
           │ Failure
           └─→ Refine 2: Even stricter instructions
                  │ Failure
                  └─→ Repair 1: Ask LLM to fix JSON
                         │ Failure
                         └─→ Repair 2: Try again with schema
                                │ Success or Failure
                                └─→ Stop
```

**Use when:** You want the best success rate (recommended)

**Code:**
```python
from patterns.patterns.iev.retry_strategies import CompositeRetryStrategy

# Default (best choice for most cases)
strategy = CompositeRetryStrategy()

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    retry_strategy=strategy,
    max_extraction_retries=4,  # 2 refinements + 2 repairs
)
```

**Or customize:**
```python
from patterns.patterns.iev.retry_strategies import (
    CompositeRetryStrategy,
    PromptRefinementRetry,
    LLMAssistedRepairRetry,
)

strategy = CompositeRetryStrategy(
    refinement_strategy=PromptRefinementRetry(max_retries=2),
    repair_strategy=LLMAssistedRepairRetry(max_retries=3),
)
```

**Pros:**
- ✅ Best success rate
- ✅ Tries cheap approach first (refinement)
- ✅ Falls back to expensive approach (repair) only if needed
- ✅ Works with all LLM types

**Cons:**
- ❌ May use extra API calls if refinement fails

---

## Using Strategies in IEV

### Basic Usage

```python
from patterns.core import create_llm_client
from patterns.patterns.iev.graph import create_iev_graph
from patterns.patterns.iev.retry_strategies import CompositeRetryStrategy
from pydantic import BaseModel

class DealInfo(BaseModel):
    company_name: str
    deal_value: float
    deal_type: str

# Setup
llm = create_llm_client(provider="openai")
strategy = CompositeRetryStrategy()  # ← Default: refinement + repair

# Create IEV graph
graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    intelligence_prompt="Analyze: {input}",
    extraction_prompt="Extract: {analysis}",
    verification_prompt="Verify: {extracted}",
    retry_strategy=strategy,  # ← Retry on JSON failures
    max_extraction_retries=4,  # ← Up to 4 attempts
)

# Run extraction
result = await graph("Acme acquired TechCorp for $1M")

# Check result
if result["verification_status"] == "APPROVED":
    print(f"Extraction attempts: {result['extraction_attempts']}")
    print(f"Deal: {result['validated']}")
else:
    print(f"Failed after retries: {result['error']}")
```

### Metrics: How Many Retries Happened?

```python
result = await graph(input_text)

print(f"Extraction attempts: {result['extraction_attempts']}")
print(f"Metrics: {result['metrics']}")

# Output:
# Extraction attempts: 2  ← First attempt failed, second retry succeeded
# Metrics: {
#   'extraction': {
#     'count': 1,
#     'avg_ms': 2543.2,
#     'success_rate': 1.0,
#     'details': {'attempts': 2, 'retry_strategy': 'CompositeRetry...'}
#   },
#   ...
# }
```

---

## Creating Custom Strategies

Implement the `IRetryStrategy` interface:

```python
from patterns.patterns.iev.retry_strategies import IRetryStrategy
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
from ...core.llm_client import LLMClient

class MyCustomRetry(IRetryStrategy):
    """Custom retry logic."""
    
    async def retry(
        self,
        llm: LLMClient,
        original_prompt: str,
        failed_output: str,
        error_message: str,
        schema: Type[BaseModel],
        attempt_number: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract valid JSON after failure.
        
        Returns:
            Valid dict matching schema, or None if retry failed
        """
        
        # Example: Use schema to guide extraction
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        
        new_prompt = f"""
        The previous extraction failed with:
        Error: {error_message}
        Output: {failed_output}
        
        Try again. This is the required schema:
        {schema_json}
        
        IMPORTANT: Return ONLY valid JSON matching this schema.
        """
        
        try:
            response = await llm.generate(
                system="You are a JSON extraction expert.",
                user=new_prompt,
            )
            
            # Parse response
            parsed = json.loads(response)
            schema.model_validate(parsed)  # Validate
            
            return parsed  # Success!
        
        except Exception as e:
            return None  # Failure
    
    @property
    def name(self) -> str:
        return "SchemaGuidedRetry"
    
    @property
    def max_retries(self) -> int:
        return 2
```

**Use it:**
```python
strategy = MyCustomRetry()

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    retry_strategy=strategy,
)
```

---

## Retry Flow Diagram

```
┌────────────────────┐
│ START: Extract JSON from LLM │
└────────────────────┘
         │
         └───▶ Attempt 1: Generate
                   │
                   ├── JSON valid? → YES → ┌──────────┐
                   │                  │ RETURN: data │
                   └── JSON valid? → NO   └──────────┘
                      │
                      └───▶ Retry strategy
                             │
                             ├─ Refinement #1
                             │  │
                             │  ├─ Success? → RETURN: data
                             │  └─ Failure? → Continue
                             │
                             ├─ Refinement #2
                             │  │
                             │  ├─ Success? → RETURN: data
                             │  └─ Failure? → Continue
                             │
                             ├─ Repair #1
                             │  │
                             │  ├─ Success? → RETURN: data
                             │  └─ Failure? → Continue
                             │
                             ├─ Repair #2
                             │  │
                             │  ├─ Success? → RETURN: data
                             │  └─ Failure? → END
                             │
                             └──────▶ ┌──────────┐
                                      │ RETURN: error │
                                      └──────────┘
```

---

## Real-World Examples

### Example 1: Extract from Brittle Local LLM

```python
import asyncio
from patterns.core import create_llm_client
from patterns.patterns.iev.graph import create_iev_graph
from patterns.patterns.iev.retry_strategies import CompositeRetryStrategy
from pydantic import BaseModel

class ParsedData(BaseModel):
    name: str
    value: float

async def main():
    # Using local Ollama (known to produce bad JSON)
    llm = create_llm_client(
        provider="ollama",
        model="mistral:7b",
    )
    
    # Use composite strategy (best for local models)
    strategy = CompositeRetryStrategy()
    
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=ParsedData,
        intelligence_prompt="Analyze: {input}",
        extraction_prompt="Extract as JSON: {analysis}",
        verification_prompt="Verify: {extracted}",
        retry_strategy=strategy,
        max_extraction_retries=4,  # Be generous with retries for local models
    )
    
    result = await graph("John Smith value 42")
    
    if result["verification_status"] == "APPROVED":
        print(f"✅ Success after {result['extraction_attempts']} attempts")
        print(f"   Data: {result['validated']}")
    else:
        print(f"❌ Failed: {result['error']}")

asyncio.run(main())
```

### Example 2: Minimize API Calls (Cost)

```python
# If you need to minimize API calls (expensive LLMs):
from patterns.patterns.iev.retry_strategies import PromptRefinementRetry

strategy = PromptRefinementRetry(max_retries=2)  # No extra API calls

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    retry_strategy=strategy,
    max_extraction_retries=2,  # Total: 1 initial + 2 retries = 3 calls
)

# vs.

from patterns.patterns.iev.retry_strategies import LLMAssistedRepairRetry

strategy = LLMAssistedRepairRetry(max_retries=3)  # Extra API calls per retry

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    retry_strategy=strategy,
    max_extraction_retries=3,  # Total: 1 initial + 3 repairs = 4 calls
)
```

---

## Best Practices

### 1. Choose the Right Strategy

| LLM Type | Recommended Strategy | Max Retries |
|----------|----------------------|-------------|
| **GPT-4 (OpenAI)** | PromptRefinement | 2 |
| **Claude (Anthropic)** | PromptRefinement | 2 |
| **Mistral 7B (Local)** | CompositeRetry | 4 |
| **Llama 2 (Local)** | CompositeRetry | 4 |
| **Unknown** | CompositeRetry | 3 |

### 2. Set Reasonable Max Retries

```python
# ✅ Good: balanced between success and cost
max_extraction_retries=4  # ~0.5-2 seconds total

# ❌ Too low: might give up too early
max_extraction_retries=1  # Only original attempt

# ❌ Too high: expensive and slow
max_extraction_retries=10  # ~5-10 seconds
```

### 3. Monitor Metrics

```python
result = await graph(input_text)

# Check if retries are helping
metrics = result["metrics"]
if metrics["extraction"]["success_rate"] < 0.8:
    print("Warning: Low extraction success rate")
    print("Consider: Better prompts, different LLM, or custom retry strategy")

if result["extraction_attempts"] > 2:
    print(f"Required {result['extraction_attempts']} attempts")
    print("Consider: Simpler schema, clearer prompt, or different LLM")
```

### 4. Combine with Validation Strategies

```python
from patterns.patterns.iev.strategies.validation import LLMAssistedValidation

# Retry handles JSON parsing failures
retry_strategy = CompositeRetryStrategy()

# Validation handles business logic failures
validation_strategy = LLMAssistedValidation()

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    retry_strategy=retry_strategy,
    validation_strategy=validation_strategy,
)

# Now you have defense in depth:
# - Retry: Handles malformed JSON
# - Validation: Handles invalid business logic
```

---

## Summary

| Strategy | Cost | Speed | Success Rate |
|----------|------|-------|---------------|
| **None** | Free | Fast | ~60% (basic JSON repair only) |
| **PromptRefinement** | Free | Fast | ~80% |
| **LLMAssistedRepair** | High | Slow | ~90% |
| **CompositeRetry** | Medium | Medium | ~95% |

**Recommendation:** Use `CompositeRetryStrategy()` by default. It's the best balance of success rate, cost, and speed.

---

**Next:** Check out [Validation Strategies Guide](./VALIDATION_STRATEGIES.md) for handling business logic verification.
