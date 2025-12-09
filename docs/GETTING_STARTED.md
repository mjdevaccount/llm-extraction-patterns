# Getting Started

Step-by-step guide to setting up and running your first pattern.

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mjdevaccount/llm-extraction-patterns.git
   cd llm-extraction-patterns
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the package**:
   ```bash
   pip install -e ".[openai]"
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

## Running Your First Pattern

Run the IEV example:

```bash
python -m patterns.patterns.iev.example
```

This will:
1. Analyze the input text
2. Extract structured deal information
3. Verify the extracted data

## Next Steps

- Modify `src/patterns/patterns/iev/prompts.py` to customize prompts
- Read `docs/PATTERNS_OVERVIEW.md` for detailed explanations
- Write your own test in `tests/test_iev.py`

