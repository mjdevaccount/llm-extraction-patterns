# Validation Strategies Guide

This guide explains how to use validation strategies in the IEV (Intelligence-Extraction-Validation) pattern.

**Quick Summary:**
Validation strategies allow you to customize how extracted data is verified before approval. Instead of hardcoding one verification approach, you can swap strategies based on your needs.

---

## Table of Contents

1. [What is a Validation Strategy?](#what-is-a-validation-strategy)
2. [Built-in Strategies](#built-in-strategies)
3. [Using Strategies in IEV](#using-strategies-in-iev)
4. [Creating Custom Strategies](#creating-custom-strategies)
5. [SOLID Principles](#solid-principles)
6. [Real-World Examples](#real-world-examples)
7. [Troubleshooting](#troubleshooting)

---

## What is a Validation Strategy?

A validation strategy defines **how** to verify that extracted data meets your requirements:

- **What** to check (required fields, value ranges, logic constraints)
- **How** to fix problems (repair attempts, type coercion, defaults)
- **When** to give up (max retries, error thresholds)

**Without strategies (before):**
```python
# Hardcoded in IEV node:
if "APPROVED" in llm_response:
    return approved
else:
    return rejected
# ❌ Same logic for everything
```

**With strategies (after):**
```python
# Choose strategy based on use case:
graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    validation_strategy=StrictFinancialValidation(),  # Financial deals
    # or
    validation_strategy=BasicValidation(),  # Simple extraction
    # or  
    validation_strategy=LLMAssistedValidation(),  # Complex logic
)
# ✅ Different logic for different needs
```

---

## Built-in Strategies

### 1. BasicValidation (Default)

**When to use:** Simple schema validation

**What it does:**
- Verifies required fields are present
- Checks field types match schema
- No repair attempts

**Example:**
```python
from patterns.patterns.iev.strategies.validation import BasicValidation

strategy = BasicValidation()

graph = create_iev_graph(
    llm_client=llm,
    output_schema=Person,
    validation_strategy=strategy,
)
```

**Schema used:**
```python
class Person(BaseModel):
    first_name: str
    last_name: str
    age: int
```

---

### 2. LLMAssistedValidation

**When to use:** Complex business logic that requires an LLM to verify

**What it does:**
- Asks LLM to verify extracted data against custom rules
- Attempts repairs if validation fails
- Provides feedback on why data was rejected

**Example:**
```python
from patterns.patterns.iev.strategies.validation import LLMAssistedValidation

strategy = LLMAssistedValidation(
    max_retries=2,
    repair_prompt="Fix this data to pass the following rules: {rules}"
)

rules = {
    "deal_value_min": 100000,
    "deal_value_max": 10000000,
    "company_name_must_be_known": True,
}

result = await graph(input_text, validation_rules=rules)
```

---

### 3. StrictFinancialValidation

**When to use:** Financial data (deals, valuations, prices)

**What it does:**
- Validates deal structures
- Checks currency and amount ranges
- Verifies deal type is known
- Repairs with type coercion ("1M" → 1000000)

**Example:**
```python
from patterns.patterns.iev.strategies.validation import StrictFinancialValidation

strategy = StrictFinancialValidation(
    min_deal_value=50000,
    max_deal_value=500000000,
    allowed_currencies=["USD", "EUR", "GBP"],
)

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    validation_strategy=strategy,
)
```

**Validates:**
```python
class DealInfo(BaseModel):
    company_name: str          # Must not be empty
    deal_value: float          # Must be within range
    deal_type: str             # Must be known type
    currency: str = "USD"      # Must be in allowed list
```

---

## Using Strategies in IEV

### Basic Usage

```python
from patterns.core import create_llm_client
from patterns.patterns.iev.graph import create_iev_graph
from patterns.patterns.iev.strategies.validation import LLMAssistedValidation
from pydantic import BaseModel

class DealInfo(BaseModel):
    company_name: str
    deal_value: float
    deal_type: str

# 1. Create LLM client
llm = create_llm_client(provider="openai", model="gpt-4")

# 2. Choose validation strategy
strategy = LLMAssistedValidation(max_retries=2)

# 3. Create IEV graph with strategy
graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    intelligence_prompt="Analyze: {input}",
    extraction_prompt="Extract: {analysis}",
    verification_prompt="Verify: {extracted}",
    validation_strategy=strategy,  # ← Use strategy here
)

# 4. Run extraction
result = await graph("Acme acquired TechCorp for $1M")

# 5. Check result
if result["verification_status"] == "APPROVED":
    deal = result["validated"]
    print(f"Deal: {deal.company_name} for ${deal.deal_value}")
else:
    print(f"Rejected: {result['error']}")
```

### Advanced: Custom Validation Rules

```python
# Some strategies support custom rules
validation_rules = {
    "min_deal_value": 50000,
    "max_deal_value": 5000000,
    "require_company_verification": True,
    "allowed_deal_types": ["acquisition", "investment", "merger"],
}

strategy = LLMAssistedValidation(
    max_retries=3,
    repair_on_failure=True,
)

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    validation_strategy=strategy,
    # validation_rules passed to strategy.validate() inside graph
)
```

---

## Creating Custom Strategies

To create your own validation strategy, implement the `IValidationStrategy` interface:

```python
from patterns.patterns.iev.abstractions import IValidationStrategy
from typing import Any, Dict, Type, Optional
from pydantic import BaseModel

class MyCustomValidation(IValidationStrategy):
    """Custom validation for your domain."""
    
    async def validate(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
        validation_rules: Dict[str, Any],
        llm: Optional['LLMClient'] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Validate data and return result.
        
        Returns dict with:
        - validated: validated data dict (or None if invalid)
        - warnings: list of warning strings
        - repairs: dict of fields that were repaired
        - outcome: "APPROVED" or "REJECTED"
        """
        
        warnings = []
        repairs = {}
        
        # 1. Check required fields
        try:
            validated_instance = schema.model_validate(data)
        except ValidationError as e:
            return {
                "validated": None,
                "warnings": [str(e)],
                "repairs": {},
                "outcome": "REJECTED",
            }
        
        # 2. Apply custom business logic
        if "deal_value" in data:
            min_value = validation_rules.get("min_deal_value", 0)
            if data["deal_value"] < min_value:
                warnings.append(f"Deal value ${data['deal_value']} below minimum ${min_value}")
        
        # 3. Decide outcome
        if len(warnings) > 0 and not validation_rules.get("allow_warnings", False):
            return {
                "validated": None,
                "warnings": warnings,
                "repairs": repairs,
                "outcome": "REJECTED",
            }
        
        # 4. Return success
        return {
            "validated": validated_instance.model_dump(),
            "warnings": warnings,
            "repairs": repairs,
            "outcome": "APPROVED",
        }
    
    @property
    def mode_name(self) -> str:
        return "CustomValidation"
```

**Use it:**
```python
strategy = MyCustomValidation()

graph = create_iev_graph(
    llm_client=llm,
    output_schema=DealInfo,
    validation_strategy=strategy,
)
```

---

## SOLID Principles

Validation strategies follow SOLID design:

### Open/Closed Principle (OCP)
**"Open for extension, closed for modification"**

❌ **Without strategies:**
```python
# Every new validation type requires modifying IEV node
async def verification_node(state):
    if domain == "finance":
        # Financial validation
    elif domain == "healthcare":
        # Healthcare validation
    elif domain == "legal":
        # Legal validation
    # ← Add new elif for each domain!
```

✅ **With strategies:**
```python
# New validation types don't touch IEV node
if domain == "finance":
    strategy = FinancialValidation()
elif domain == "healthcare":
    strategy = HealthcareValidation()
elif domain == "legal":
    strategy = LegalValidation()

# IEV node is unchanged
graph = create_iev_graph(
    llm_client=llm,
    validation_strategy=strategy,  # ← Just swap the strategy
)
```

### Dependency Inversion Principle (DIP)
**"Depend on abstractions, not concretions"**

✅ IEV graph depends on `IValidationStrategy` interface:
```python
# IEV doesn't know about specific strategies
if validation_strategy:
    result = await validation_strategy.validate(...)  # Uses interface
```

Not on concrete implementations:
```python
# ❌ IEV would depend on ALL possible strategies
if isinstance(validation_strategy, LLMAssistedValidation):
    ...
elif isinstance(validation_strategy, FinancialValidation):
    ...
```

---

## Real-World Examples

### Example 1: Deal Extraction with Financial Validation

```python
import asyncio
from patterns.core import create_llm_client
from patterns.patterns.iev.graph import create_iev_graph
from patterns.patterns.iev.strategies.validation import StrictFinancialValidation
from pydantic import BaseModel

class DealInfo(BaseModel):
    company_name: str
    target_name: str
    deal_value: float
    deal_type: str  # "acquisition", "investment", "merger"

async def extract_deal(text: str):
    llm = create_llm_client(provider="openai")
    
    strategy = StrictFinancialValidation(
        min_deal_value=100000,
        max_deal_value=10000000,
    )
    
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=DealInfo,
        intelligence_prompt="Extract deal details from: {input}",
        extraction_prompt="Get deal info: {analysis}",
        verification_prompt="Verify deal: {extracted}",
        validation_strategy=strategy,
    )
    
    result = await graph(text)
    
    if result["verification_status"] == "APPROVED":
        return result["validated"]
    else:
        print(f"Deal rejected: {result['error']}")
        return None

# Usage
deal = asyncio.run(extract_deal(
    "Acme Inc acquired TechCorp for $2.5M in an acquisition"
))
if deal:
    print(f"✅ {deal.company_name} acquired {deal.target_name} for ${deal.deal_value}")
```

### Example 2: Custom Medical Data Validation

```python
class PatientInfo(BaseModel):
    name: str
    age: int
    diagnosis: str

class MedicalValidation(IValidationStrategy):
    async def validate(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
        validation_rules: Dict[str, Any],
        llm: Optional['LLMClient'] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        
        # Validate basic schema
        try:
            validated = schema.model_validate(data)
        except ValidationError as e:
            return {
                "validated": None,
                "outcome": "REJECTED",
                "warnings": [str(e)],
                "repairs": {},
            }
        
        # Medical-specific checks
        warnings = []
        
        # Age must be 0-150
        if not (0 <= data["age"] <= 150):
            warnings.append(f"Invalid age: {data['age']}")
        
        # Diagnosis must be from approved list
        approved_diagnoses = validation_rules.get("approved_diagnoses", [])
        if approved_diagnoses and data["diagnosis"] not in approved_diagnoses:
            warnings.append(f"Unknown diagnosis: {data['diagnosis']}")
        
        # Return result
        return {
            "validated": validated.model_dump() if not warnings else None,
            "outcome": "APPROVED" if not warnings else "REJECTED",
            "warnings": warnings,
            "repairs": {},
        }
    
    @property
    def mode_name(self) -> str:
        return "MedicalValidation"

# Use it
strategy = MedicalValidation()

graph = create_iev_graph(
    llm_client=llm,
    output_schema=PatientInfo,
    validation_strategy=strategy,
)
```

---

## Troubleshooting

### "Validation strategy not being used"

**Check:**
1. Is `validation_strategy` parameter passed to `create_iev_graph`?
   ```python
   graph = create_iev_graph(
       llm_client=llm,
       validation_strategy=my_strategy,  # ← Must be here
   )
   ```

2. Is your strategy implemented correctly?
   ```python
   # Must implement IValidationStrategy
   class MyStrategy(IValidationStrategy):
       async def validate(...):  # Must be async
           pass
       
       @property
       def mode_name(self) -> str:
           return "MyStrategy"
   ```

### "Validation always rejects"

**Check:**
1. Does `validate()` return the correct dict structure?
   ```python
   return {
       "validated": data_dict_or_none,
       "outcome": "APPROVED" or "REJECTED",  # ← Must be exact string
       "warnings": list,
       "repairs": dict,
   }
   ```

2. Is `outcome` spelled correctly? (Common typo: "PASSED" vs "APPROVED")

### "Why is LLMAssistedValidation slow?"

It makes an extra LLM call for each validation. Use `BasicValidation` if you don't need LLM verification:
```python
strategy = BasicValidation()  # Just schema validation, fast
```

---

## Summary

| Strategy | Use When | Speed | Complexity |
|----------|----------|-------|-------------|
| **None** | Simple schema validation only | Fast | Low |
| **BasicValidation** | Straightforward extraction | Fast | Low |
| **LLMAssistedValidation** | Complex business logic | Slow | High |
| **StrictFinancialValidation** | Financial data | Medium | Medium |
| **Custom** | Domain-specific needs | Varies | Varies |

---

**Next:** Check out [Retry Strategies Guide](./RETRY_STRATEGIES.md) for handling extraction failures.
