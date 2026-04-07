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

| Argument             | Default                        | Description                              |
|----------------------|--------------------------------|------------------------------------------|
| `--model`            | `groq/llama-3.1-8b-instant`   | LLM for JSON generation and extraction   |
| `--text-gen-model`   | same as `--model`              | LLM for synthetic text generation        |
| `--optimizer-model`  | same as `--model`              | LLM for prompt refinement                |

### Hyperparameters

| Argument              | Default | Description                           |
|-----------------------|---------|---------------------------------------|
| `--batch-size`        | 10      | Synthetic examples per step           |
| `--max-steps`         | 10      | Maximum optimization steps            |
| `--min-steps`         | 0       | Minimum steps before early stopping   |
| `--temp-json`         | 0.5     | Temperature for JSON generation       |
| `--temp-extract`      | 0.0     | Temperature for JSON extraction       |
| `--temp-article`      | 0.0     | Temperature for text generation       |

### Stopping criteria

| Argument                  | Default | Description                          |
|---------------------------|---------|--------------------------------------|
| `--field-overlap-target`  | 0.99    | Stop when field overlap exceeds this |
| `--json-distance-target`  | 0.01    | Stop when JSON distance drops below  |
| `--schema-valid-target`   | 0.99    | Minimum schema validity rate         |
| `--rollback-threshold`    | 0.01    | Score drop that triggers rollback    |

### Output options

| Argument            | Default                      | Description                     |
|---------------------|------------------------------|---------------------------------|
| `--output`, `-o`    | `optimized_prompt.txt`       | Output file for optimized prompt|
| `--db-path`         | `optimize_prompt_json.db`    | SQLite database for run history |
| `--log-dir`         | `logs`                       | Directory for log files         |
| `--quiet`, `-q`     | off                          | Suppress step-by-step output    |
| `--no-optimize`     | off                          | Run baseline only               |

## Example with all options

```bash
optimize-prompt-json \
  --schema my_schema.json \
  --text my_article.txt \
  --model groq/llama-3.1-8b-instant \
  --text-gen-model groq/openai/gpt-oss-120b \
  --optimizer-model groq/openai/gpt-oss-120b \
  --batch-size 10 \
  --max-steps 10 \
  --temp-json 0.5 \
  --output my_optimized_prompt.txt
```

## Supported LLM providers

Any model supported by [litellm](https://docs.litellm.ai/docs/providers) works. Set the corresponding API key in `.env`:

| Provider   | Env variable        | Example model                                  |
|------------|---------------------|-------------------------------------------------|
| Groq       | `GROQ_API_KEY`      | `groq/llama-3.1-8b-instant`                    |
| OpenAI     | `OPENAI_API_KEY`    | `gpt-4.1-nano`                                 |
| DeepSeek   | `DEEPSEEK_API_KEY`  | `deepseek/deepseek-chat`                        |
| Google     | `GEMINI_API_KEY`    | `gemini/gemini-2.5-flash-lite`                  |
| Mistral    | `MISTRAL_API_KEY`   | `mistral/mistral-small`                         |

## Library usage

You can also use `optimize-prompt-json` as a Python library in your own code:

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

This allows you to integrate prompt optimization into your own pipelines or applications.

## Output

The tool produces:

- **Console output**: Step-by-step progress, quality comparison (step 0 vs final), and the optimized prompt
- **`optimized_prompt.txt`**: The refined extraction prompt ready for production use
- **`optimize_prompt_json.db`**: SQLite database with full run history and metrics
- **`logs/`**: Detailed log files for debugging

## License

MIT
