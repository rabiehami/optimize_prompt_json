"""Command-line interface for optimize-prompt-json."""

import argparse
import asyncio
import json
import sys

from optimize_prompt_json import __version__


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="optimize-prompt-json",
        description="Iterative LLM prompt optimization for JSON extraction from text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic usage
  optimize-prompt-json --schema schema.json --text article.txt

  # Specify model
  optimize-prompt-json --schema schema.json --text article.txt \\
    --model groq/llama-3.1-8b-instant

  # Full configuration
  optimize-prompt-json --schema schema.json --text article.txt \\
    --model groq/llama-3.1-8b-instant \\
    --text-gen-model groq/openai/gpt-oss-120b \\
    --optimizer-model groq/openai/gpt-oss-120b \\
    --batch-size 10 --max-steps 10 --temp-json 0.5
""",
    )

    # Required
    parser.add_argument("--schema", required=True, help="Path to JSON schema file")
    parser.add_argument("--text", required=True, help="Path to text file to extract JSON from")

    # Model configuration
    parser.add_argument(
        "--model", default="groq/llama-3.1-8b-instant",
        help="LLM model for JSON generation and extraction (default: groq/llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--text-gen-model", default=None,
        help="LLM model for synthetic text generation (default: same as --model)",
    )
    parser.add_argument(
        "--optimizer-model", default=None,
        help="LLM model for prompt refinement and lessons (default: same as --model)",
    )

    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size per step (default: 10)")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum optimization steps (default: 10)")
    parser.add_argument("--min-steps", type=int, default=0, help="Minimum optimization steps (default: 0)")
    parser.add_argument("--temp-json", type=float, default=0.5, help="Temperature for JSON generation (default: 0.5)")
    parser.add_argument("--temp-extract", type=float, default=0.0, help="Temperature for JSON extraction (default: 0.0)")
    parser.add_argument("--temp-article", type=float, default=0.0, help="Temperature for text generation (default: 0.0)")

    # Stopping criteria
    parser.add_argument("--field-overlap-target", type=float, default=0.99)
    parser.add_argument("--json-distance-target", type=float, default=0.01)
    parser.add_argument("--schema-valid-target", type=float, default=0.99)
    parser.add_argument("--rollback-threshold", type=float, default=0.01)

    # Rate limiting
    parser.add_argument("--rate-limit-delay", type=float, default=0.0, help="Delay between API requests in seconds")

    # Output
    parser.add_argument("--output", "-o", default="optimized_prompt.txt", help="Output file for optimized prompt")
    parser.add_argument("--db-path", default="optimize_prompt_json.db", help="SQLite database path")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress step-by-step console output")
    parser.add_argument("--no-optimize", action="store_true", help="Run baseline only without optimization")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry point."""
    args = parse_args(argv)

    # Read schema
    try:
        with open(args.schema, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError:
        print(f"Error: Schema file not found: {args.schema}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in schema file: {e}", file=sys.stderr)
        sys.exit(1)

    # Read text
    try:
        with open(args.text, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Text file not found: {args.text}", file=sys.stderr)
        sys.exit(1)

    if not text.strip():
        print("Error: Text file is empty", file=sys.stderr)
        sys.exit(1)

    # Build config
    from optimize_prompt_json.pipeline import OptimizationConfig, run_optimization

    config = OptimizationConfig(
        schema=schema,
        text=text,
        schema_path=args.schema,
        text_path=args.text,
        llm_model=args.model,
        llm_text_gen_model=args.text_gen_model,
        llm_optimizer_model=args.optimizer_model,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        temp_json=args.temp_json,
        temp_extract=args.temp_extract,
        temp_article=args.temp_article,
        field_overlap_target=args.field_overlap_target,
        json_distance_target=args.json_distance_target,
        schema_valid_target=args.schema_valid_target,
        rollback_threshold=args.rollback_threshold,
        rate_limit_delay=args.rate_limit_delay,
        optimize=not args.no_optimize,
        output_path=args.output,
        db_url=f"sqlite:///{args.db_path}",
        log_dir=args.log_dir,
        quiet=args.quiet,
    )

    result = asyncio.run(run_optimization(config))
    sys.exit(0 if result else 1)
