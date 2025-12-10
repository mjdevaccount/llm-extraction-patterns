# Two-Model Pattern

## Overview

The two-model pattern separates **reasoning** from **formatting** by using two specialized LLMs:

1. **Main LLM** (large, smart) - Generates unstructured analysis
2. **Extractor LLM** (small, specialized) - Converts to structured JSON

## Pattern Structure

```
Main LLM → Unstructured Text → Extractor Formatter → Extractor LLM → Structured JSON
```

## Key Components

### 1. `IExtractorFormatter` (Abstraction)

The **mapping interface** that handles the transformation between:
- Main LLM output (unstructured text)
- Extractor LLM input (formatted prompt)

**Responsibilities:**
- Format unstructured text + schema into extractor prompt
- Extract JSON from extractor response
- Provide extractor configuration recommendations

### 2. `NuExtractFormatter` (Implementation)

Concrete implementation for NuExtract model format:

```python
formatter = NuExtractFormatter()

# Formats to NuExtract's expected format:
# <|input|>
# ### Template:
# {schema_json}
# ### Text:
# {unstructured_text}
# <|output|>
```

### 3. `TwoModelExtractionNode` (Orchestrator)

Node that coordinates the two-model workflow:

```python
node = TwoModelExtractionNode(
    main_llm=main_model,           # Large model for reasoning
    extractor_llm=extractor_model, # Small model for extraction
    extractor_formatter=formatter, # Handles the mapping
    output_schema=ValuationResult,
    main_prompt_template="Analyze: {input}",
    required_state_keys=["input"]
)
```

## Design Principles

### Single Responsibility
- **IExtractorFormatter**: Only handles mapping/formatting
- **TwoModelExtractionNode**: Only orchestrates the workflow
- **NuExtractFormatter**: Only implements NuExtract format

### Open/Closed Principle
- New extractor formats can be added by implementing `IExtractorFormatter`
- No need to modify `TwoModelExtractionNode` when adding new formatters

### Dependency Inversion
- Node depends on `IExtractorFormatter` abstraction, not concrete implementations
- Formatters can be swapped without changing node logic

## Usage Example

```python
from patterns.patterns.two_model import TwoModelExtractionNode, NuExtractFormatter

# Create formatter (handles the mapping)
formatter = NuExtractFormatter()

# Create node
node = TwoModelExtractionNode(
    main_llm=main_model,
    extractor_llm=extractor_model,
    extractor_formatter=formatter,
    output_schema=MySchema,
    main_prompt_template="Analyze: {input}",
    required_state_keys=["input"]
)

# Execute
state = await node.execute({"input": "..."})
result = state["extracted"]  # Structured JSON
```

## Benefits

1. **Cost Efficiency**: Use large model only for reasoning
2. **VRAM Efficiency**: Can fit both models on 16GB GPU
3. **Separation of Concerns**: Reasoning vs. formatting
4. **Flexibility**: Swap extractor formats without changing workflow

## Adding New Extractor Formats

To add support for a new extractor model:

1. Implement `IExtractorFormatter`:

```python
class MyExtractorFormatter(IExtractorFormatter):
    def format_prompt(self, unstructured_text, schema, max_length):
        # Your formatting logic
        return formatted_prompt
    
    def extract_json_from_response(self, extractor_response):
        # Your extraction logic
        return json_string
    
    def get_extractor_config(self):
        # Recommended config for your extractor
        return {"temperature": 0.0, ...}
    
    @property
    def name(self):
        return "MyExtractor"
```

2. Use it:

```python
formatter = MyExtractorFormatter()
node = TwoModelExtractionNode(..., extractor_formatter=formatter)
```

No changes needed to `TwoModelExtractionNode`!

