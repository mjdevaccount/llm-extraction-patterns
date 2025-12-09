# Architecture

Design decisions and architectural overview.

## Core Principles

1. **Modularity**: Each pattern is independent and can be understood in isolation
2. **Simplicity**: Minimal dependencies, clear code structure
3. **Learnability**: Patterns build on each other progressively
4. **Flexibility**: Easy to swap LLM providers, add tools, extend patterns

## Design Decisions

### Why LangGraph?

- Provides structured state machines (better than ad-hoc loops)
- Built-in persistence & replay
- Native integration with LangChain ecosystem

### Why MCP-First Tool Integration?

- Your aistack-mcp is the source of truth
- `core/mcp_tools.py` wraps it for easy access
- Each pattern can call tools without reinventing

### Why One Example Per Pattern?

- Not 50 examples. Just 1 runnable, clear example per pattern.
- Easy to understand and modify.

### Why Type Safety with Pydantic?

- All inputs/outputs have type hints
- Easier to debug and extend

## File Organization

```
src/patterns/
├── core/           # Shared infrastructure (LLM, MCP, memory)
├── patterns/       # Pattern implementations
└── utils/          # Learning & debugging tools
```

Each pattern follows the same structure:
- `graph.py`: State machine definition
- `prompts.py`: Prompt templates
- `example.py`: Minimal working example

