# claude-eval

A lightweight evaluation harness for LLM outputs, built on the Claude API.

Define test cases in YAML → run them against Claude → score with LLM-as-judge → get a structured report.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Why

Building with LLMs is easy. Knowing whether your prompts *actually work* is hard. `claude-eval` gives you a repeatable, version-controlled way to measure prompt quality — so you can iterate with confidence instead of vibes.

**Use cases:**
- Compare prompt variants ("which system prompt is better?")
- Regression-test prompts before deploying changes
- Benchmark different models on your specific task
- Build a scored dataset to demonstrate LLM behavior

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/claude-eval.git
cd claude-eval
pip install -e .

# 2. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run the example eval
claude-eval run evals/summarization.yaml
```

## How It Works

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌────────────┐
│  YAML eval   │────▶│  Run against  │────▶│  LLM-as-     │────▶│  JSON/CLI  │
│  definition  │     │  Claude API   │     │  Judge scores │     │  report    │
└──────────────┘     └───────────────┘     └──────────────┘     └────────────┘
```

### 1. Define an eval (`evals/summarization.yaml`)

```yaml
name: summarization_quality
description: Test whether Claude produces concise, accurate summaries
model: claude-sonnet-4-20250514
system_prompt: "Summarize the following text in 2-3 sentences. Be concise and accurate."

judge:
  criteria:
    - name: accuracy
      description: "Does the summary capture the key facts without hallucination?"
      weight: 0.4
    - name: conciseness
      description: "Is the summary 2-3 sentences and free of filler?"
      weight: 0.3
    - name: completeness
      description: "Does it cover the most important points?"
      weight: 0.3

cases:
  - input: |
      The James Webb Space Telescope launched on December 25, 2021.
      It is the largest optical telescope in space, with a 6.5-meter
      primary mirror. JWST orbits the Sun near the Earth-Sun L2 point,
      about 1.5 million km from Earth. Its primary goals include
      observing the first galaxies formed after the Big Bang and
      characterizing potentially habitable exoplanets.
    expected_keywords: ["James Webb", "telescope", "2021"]
    
  - input: |
      Python 3.12 introduced several performance improvements including
      a new type system for the interpreter. The release also added
      f-string improvements allowing nested quotes and backslashes.
      PEP 709 introduced comprehension inlining for better performance.
      The release was made available on October 2, 2023.
    expected_keywords: ["Python 3.12", "performance", "2023"]
```

### 2. Run it

```bash
# Default run
claude-eval run evals/summarization.yaml

# With options
claude-eval run evals/summarization.yaml --output results/run_001.json --verbose
```

### 3. Read the report

```
═══════════════════════════════════════════
  claude-eval results: summarization_quality
═══════════════════════════════════════════

  Model:  claude-sonnet-4-20250514
  Cases:  2
  Time:   3.4s

  ┌────────────────┬───────┬────────┐
  │ Criteria       │ Avg   │ Weight │
  ├────────────────┼───────┼────────┤
  │ accuracy       │ 4.5/5 │  0.4   │
  │ conciseness    │ 5.0/5 │  0.3   │
  │ completeness   │ 4.0/5 │  0.3   │
  └────────────────┴───────┴────────┘

  Weighted Score: 4.45 / 5.0 (89.0%)
```

## Project Structure

```
claude-eval/
├── claude_eval/
│   ├── __init__.py
│   ├── cli.py          # CLI entry point
│   ├── runner.py       # Runs eval cases against Claude API
│   ├── judge.py        # LLM-as-judge scoring logic
│   └── report.py       # Formats and outputs results
├── evals/
│   └── summarization.yaml   # Example eval definition
├── results/                 # Output directory for run results
├── tests/
│   └── test_judge.py        # Unit tests
├── pyproject.toml
└── README.md
```

## Writing Custom Evals

Create a YAML file in `evals/` with:

| Field | Required | Description |
|---|---|---|
| `name` | ✓ | Identifier for the eval |
| `model` | ✓ | Claude model to test |
| `system_prompt` | ✓ | The prompt you're evaluating |
| `judge.criteria` | ✓ | What to score on (name, description, weight) |
| `cases` | ✓ | List of test inputs with optional `expected_keywords` |
| `description` | | Human-readable description |

## Tech Stack

- **Python 3.10+** — modern, clean async
- **Anthropic SDK** — official Claude API client
- **PyYAML** — eval definitions
- **Click** — CLI framework
- **LLM-as-Judge pattern** — uses Claude to score Claude (configurable criteria)

## License

MIT
