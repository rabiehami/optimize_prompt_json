# optimize-prompt-json

Iterative LLM prompt optimization for JSON extraction from text.

Given a JSON schema and a sample text, this tool automatically refines the extraction prompt through iterative optimization — generating synthetic training data, evaluating extraction quality, and improving the prompt step by step.

## How it works

1. **Generate** random JSON instances from your schema
2. **Transform** each JSON into natural-language text (using your sample as a style reference)
3. **Extract** JSON back from the synthetic text using the current prompt
4. **Evaluate** extraction quality (field overlap, value similarity, schema validity)
5. **Refine** the prompt based on lessons learned from errors
6. **Repeat** until quality targets are met or max steps reached

The output is an **optimized extraction prompt** that you can use with any LLM to extract structured JSON from text matching your schema.

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```


## Quick start

1. Copy `.env.example` to `.env` and add your API key(s):
  ```
  GROQ_API_KEY=your_key_here
  ```

2. Use the library in your Python code:
  ```python
  from optimize_prompt_json import OptimizationConfig, run_optimization
  import asyncio

  config = OptimizationConfig(
     schema={"type": "object", "properties": {"foo": {"type": "string"}}},
     text="Sample text to extract from."
  )
  result = asyncio.run(run_optimization(config))
  print(result["optimized_prompt"])
  ```

The optimized prompt is available in the result dictionary.


## Output

The tool produces:

- **Console output**: Step-by-step progress, quality comparison (step 0 vs final), and the optimized prompt
- **`optimized_prompt.txt`**: The refined extraction prompt ready for production use
- **`optimize_prompt_json.db`**: SQLite database with full run history and metrics
- **`logs/`**: Detailed log files for debugging

## Supported LLM providers

Any model supported by [litellm](https://docs.litellm.ai/docs/providers) works.

**Recommended:** Groq models (e.g., `groq/llama-3.1-8b-instant`) are a good choice due to their high rate limits and performance.

## License

MIT
