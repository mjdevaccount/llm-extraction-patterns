# LLM Design Patterns Repository - Foundation

A modular, learning-first repository for implementing and understanding five core LLM agent design patterns. Built with Python, LangGraph, and MCP integration.

## Repository Structure

```
aistack-patterns/
│
├── README.md                    # This file
├── pyproject.toml              # Project metadata & dependencies
├── .env.example                # Configuration template
│
├── src/
│   └── patterns/
│       │
│       ├── __init__.py         # Package exports
│       │
│       ├── core/               # ✅ SHARED INFRASTRUCTURE
│       │   ├── __init__.py
│       │   ├── llm_client.py   # LLM wrapper (OpenAI, Anthropic, local Ollama)
│       │   ├── mcp_tools.py    # Your aistack-mcp connector
│       │   ├── memory.py       # Vector DB + context management
│       │   ├── types.py        # Shared type definitions
│       │   └── logger.py       # Logging utilities
│       │
│       ├── patterns/           # ✅ THE 5 PATTERN IMPLEMENTATIONS
│       │   ├── __init__.py
│       │   │
│       │   ├── iev/            # Pattern 1: Intelligence-Extraction-Validation
│       │   │   ├── __init__.py
│       │   │   ├── graph.py    # State machine definition
│       │   │   ├── prompts.py  # Verification-specific prompts
│       │   │   └── example.py  # Minimal working example
│       │   │
│       │   ├── evaluator_optimizer/  # Pattern 2: Draft-Critique-Refine
│       │   │   ├── __init__.py
│       │   │   ├── graph.py
│       │   │   ├── grader.py   # Scoring & feedback logic
│       │   │   └── example.py
│       │   │
│       │   ├── orchestrator/        # Pattern 3: Orchestrator-Workers
│       │   │   ├── __init__.py
│       │   │   ├── orchestrator.py  # Router logic
│       │   │   ├── workers.py       # Worker definitions
│       │   │   └── example.py
│       │   │
│       │   ├── agentic_rag/         # Pattern 4: Iterative Retrieval-Verification
│       │   │   ├── __init__.py
│       │   │   ├── retriever.py     # Search logic
│       │   │   ├── verifier.py      # Sufficiency checker
│       │   │   └── example.py
│       │   │
│       │   └── system2/             # Pattern 5: Thinking-Before-Acting
│       │       ├── __init__.py
│       │       ├── flow.py          # <thought> generation pipeline
│       │       ├── parser.py        # Extract reasoning
│       │       └── example.py
│       │
│       └── utils/             # ✅ LEARNING & DEBUGGING
│           ├── __init__.py
│           ├── test_helpers.py     # Mock LLMs for testing
│           ├── visualizer.py       # Graph state visualization
│           └── prompt_tester.py    # Quick prompt iteration
│
├── tests/                      # Unit tests (one per pattern)
│   ├── __init__.py
│   ├── test_iev.py
│   ├── test_evaluator_opt.py
│   ├── test_orchestrator.py
│   ├── test_agentic_rag.py
│   └── test_system2.py
│
├── docs/                       # Learning materials
│   ├── PATTERNS_OVERVIEW.md    # Detailed pattern explanations
│   ├── GETTING_STARTED.md      # Step-by-step setup
│   ├── ARCHITECTURE.md         # Design decisions
│   └── examples/               # Real-world use cases
│
├── examples/                   # Runnable demonstrations
│   ├── simple_iev.py           # "Delete this file safely"
│   ├── code_review.py          # Evaluator-Optimizer for code
│   ├── research_project.py     # Orchestrator for multi-step project
│   ├── fact_checker.py         # Agentic RAG example
│   └── math_problem.py         # System 2 for complex logic
│
├── .gitignore
├── .env.example                # Copy to .env for local config
└── Makefile                    # Common commands (run, test, docs)
```

## Core Philosophy: "Learn One Pattern at a Time"

Each pattern is independent and minimal. You can understand Pattern 1 (IEV) completely before touching Pattern 2.

## Getting Started (3 Steps)

### Step 1: Choose Your First Pattern

Start with IEV (Intelligence-Extraction-Validation) because:

✅ Simplest mental model: Explore → Verify → Act

✅ Direct connection to safety (most intuitive)

✅ Only 3 nodes in the graph

### Step 2: Run the Minimal Example

```bash
# Install dependencies
pip install -e ".[openai]"

# Copy environment template
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run the IEV example
python -m patterns.patterns.iev.example
```

This executes a simple scenario like: "Extract deal information from text and verify it."

### Step 3: Modify & Learn

Open `src/patterns/patterns/iev/example.py` and:

- Change the verification prompt
- Add a new tool call
- See how the graph behaves

## Pattern Quick Reference

| Pattern | File | Purpose | Complexity | When to Use |
|---------|------|---------|------------|-------------|
| IEV | `patterns/iev/` | Safety & Precision | ⭐ Easy | High-stakes actions |
| Evaluator-Optimizer | `patterns/evaluator_optimizer/` | Quality Control | ⭐⭐ Medium | Content refinement |
| Orchestrator | `patterns/orchestrator/` | Complex Tasks | ⭐⭐⭐ Medium | Multi-step projects |
| Agentic RAG | `patterns/agentic_rag/` | Research & Lookup | ⭐⭐ Medium | Fact-heavy queries |
| System 2 | `patterns/system2/` | Hard Logic | ⭐⭐⭐ Medium | Math/reasoning |

## Core Module Descriptions

### `core/llm_client.py`

Unified interface to swap LLM providers:

```python
from patterns.core import create_llm_client

# Use OpenAI
llm = create_llm_client(provider="openai", model="gpt-4")

# Or local Ollama
llm = create_llm_client(provider="ollama", model="mistral:7b")

# Call the same way regardless
response = llm.generate(system="You are helpful", user="Hello")
```

### `core/mcp_tools.py`

Your aistack-mcp integration point:

```python
from patterns.core import get_mcp_toolkit

tools = get_mcp_toolkit()

# Use in any pattern
result = tools.call("file_read", path="/path/to/file")
```

### `core/memory.py`

Short-term context management:

```python
from patterns.core import ConversationMemory

memory = ConversationMemory(max_history=10)
memory.add_message(role="user", content="...")
memory.add_message(role="assistant", content="...")

# Used by any pattern to maintain state
context = memory.get_context()
```

## Dependency Stack

```toml
[project]
dependencies = [
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.24.0",
]

[project.optional-dependencies]
openai = ["openai>=1.3.0"]
anthropic = ["anthropic>=0.25.0"]
ollama = ["ollama>=0.1.0"]
langgraph = ["langgraph>=0.2.0", "langchain>=0.1.0"]
dev = ["pytest>=7.4.0", "black>=23.0.0", "ruff>=0.1.0"]
```

## Learning Path

**Week 1: Foundation**
- Read `docs/PATTERNS_OVERVIEW.md`
- Run `python -m patterns.patterns.iev.example`
- Modify the verification prompt
- Add your own tool call

**Week 2: Deepen IEV**
- Understand the state transitions
- Write a test in `tests/test_iev.py`
- Build a custom scenario (e.g., "Verify before trading")

**Week 3: Layer Evaluator-Optimizer**
- Run examples/code_review.py
- Understand the critique loop
- Experiment with grading criteria

**Week 4+: Combine Patterns**
- Use Orchestrator to delegate tasks
- Integrate Agentic RAG for research
- Implement System 2 for logic puzzles

## Running Locally

### Prerequisites

- Python 3.10+
- Your aistack-mcp server running (optional, for MCP tests)

### Setup

```bash
# Clone the repo
git clone https://github.com/mjdevaccount/llm-extraction-patterns.git
cd llm-extraction-patterns

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[openai]"

# Copy environment template
cp .env.example .env

# Edit .env with your API keys (OpenAI, Anthropic, etc.)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Run the first example
python -m patterns.patterns.iev.example
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Different LLM Providers

```bash
# Using Anthropic Claude
PROVIDER=anthropic python -m patterns.patterns.iev.example

# Using local Ollama
PROVIDER=ollama MODEL=mistral:7b python -m patterns.patterns.iev.example

# Using OpenAI (default)
PROVIDER=openai python -m patterns.patterns.iev.example
```

## What's NOT in This Repo

❌ Pre-built production agents (this is for learning, not deployment)

❌ Complex multi-repo orchestration (keep it simple)

❌ Web UI or API (focus on patterns, not plumbing)

❌ Financial-specific logic (patterns are domain-agnostic)

## Next Steps

1. Clone & setup this repo (5 min)
2. Run the IEV example (5 min)
3. Read the IEV explanation in `docs/PATTERNS_OVERVIEW.md` (20 min)
4. Modify the prompt in `patterns/iev/prompts.py` (10 min)
5. Write a test for your scenario (15 min)

Once you've mastered IEV, move to Evaluator-Optimizer (same process).

## Contributing & Extending

To add your own pattern or example:

1. Create a new folder in `src/patterns/patterns/{pattern_name}/`
2. Include: `__init__.py`, `graph.py`, `example.py`, `prompts.py`
3. Add a test in `tests/test_{pattern_name}.py`
4. Update `docs/PATTERNS_OVERVIEW.md` with an explanation
5. Document the learning path

## License

MIT

## Questions & Support

- GitHub Issues: For bugs and feature requests
- Documentation: Check `docs/` for detailed explanations

**Start with IEV. Master it. Then layer on the others.**
