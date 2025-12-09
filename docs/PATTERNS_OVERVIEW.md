# Patterns Overview

Detailed explanations of each LLM design pattern.

## Pattern 1: IEV (Intelligence-Extraction-Validation)

**Status**: ✅ Implemented

The IEV pattern ensures safety and precision by following three phases:

1. **Intelligence**: Analyze the input to understand context and requirements
2. **Extraction**: Extract structured data matching a schema
3. **Validation**: Verify the extracted data is correct and safe

**Use Cases**:
- Extracting deal information from emails
- Parsing structured data from unstructured text
- Any scenario requiring verification before action

## Pattern 2: Evaluator-Optimizer

**Status**: 🚧 Coming Soon

Draft-Critique-Refine loop for quality improvement.

## Pattern 3: Orchestrator

**Status**: 🚧 Coming Soon

Delegates complex tasks to specialized workers.

## Pattern 4: Agentic RAG

**Status**: 🚧 Coming Soon

Iterative retrieval and verification for fact-heavy queries.

## Pattern 5: System 2

**Status**: 🚧 Coming Soon

Thinking-before-acting for complex reasoning tasks.

