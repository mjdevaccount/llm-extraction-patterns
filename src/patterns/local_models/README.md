# Local Model Patterns

This directory contains advanced patterns and utilities specifically designed for **local models** (Ollama, etc.) that require more hand-holding and sophisticated error handling.

## Why Separate?

Local models often have:
- Lower reliability than cloud models
- Need more sophisticated JSON repair strategies
- Require retry logic and fallback mechanisms
- Benefit from advanced validation strategies
- Need workflow orchestration with better error recovery

## Structure

```
local_models/
├── strategies/          # Advanced JSON repair and validation strategies
├── workflows/           # Advanced workflow orchestration
├── nodes/              # Enhanced nodes with local model support
└── helpers/            # Helper workflows (tool-calling, conditional, etc.)
```

## Usage

These patterns are **optional** - the simpler patterns in `patterns/` work fine for cloud models. Use these when:

1. Working with local models (Ollama, etc.)
2. Need advanced JSON repair
3. Require sophisticated validation strategies
4. Want workflow orchestration with error recovery

## SOLID Principles

This codebase follows SOLID principles strictly:
- **Single Responsibility**: Each class/strategy does ONE thing
- **Open/Closed**: Extend via strategies, not code changes
- **Liskov Substitution**: All implementations are interchangeable
- **Interface Segregation**: Minimal required interfaces
- **Dependency Inversion**: Depend on abstractions only

**No legacy code or backward compatibility fallbacks** - pure SOLID design.

