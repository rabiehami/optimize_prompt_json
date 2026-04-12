"""Console output formatting for optimization progress and results."""

import math
import os


def print_header(config):
    """Print startup configuration summary."""
    n_props = len(config.schema.get("properties", {}))
    text_len = len(config.text)
    text_gen = config.llm_text_gen_model or config.llm_model
    optimizer = config.llm_optimizer_model or config.llm_model

    print()
    print("=" * 64)
    print("  optimize-prompt-json v0.1.0")
    print("=" * 64)
    print(f"  Schema:      {os.path.basename(config.schema_path)} ({n_props} properties)")
    print(f"  Text:        {os.path.basename(config.text_path)} ({text_len:,} chars)")
    print(f"  Model:       {config.llm_model}")
    if text_gen != config.llm_model:
        print(f"  Text gen:    {text_gen}")
    if optimizer != config.llm_model:
        print(f"  Optimizer:   {optimizer}")
    print(f"  Batch size:  {config.batch_size}   |  Max steps: {config.max_steps}")
    print("=" * 64)
    print()


def print_step(step_id, max_steps, score, prev_score, metrics, rolled_back=False):
    """Print single-line step progress."""
    if metrics is None:
        print(f"  [Step {step_id}/{max_steps}] -- No metrics (all artifacts invalid)")
        return

    delta_str = ""
    if prev_score is not None and prev_score >= 0:
        delta = score - prev_score
        sign = "+" if delta >= 0 else ""
        delta_str = f" ({sign}{delta:.3f})"

    rollback_str = " << Rolled back to best prompt" if rolled_back else ""

    valid = metrics.get("schema_valid_rate", 0) * 100
    overlap = metrics.get("field_overlap_mean", 0)
    sim = metrics.get("value_similarity_mean", 0)
    dist = metrics.get("json_distance_mean", 0)

    print(
        f"  [Step {step_id}/{max_steps}] Score: {score:.4f}{delta_str}{rollback_str}\n"
        f"    Valid: {valid:.0f}%  Overlap: {overlap:.3f}  "
        f"Similarity: {sim:.3f}  Distance: {dist:.3f}"
    )


def print_early_stop(step_id):
    """Print early stopping notice."""
    print(f"\n  >> Early stopping at step {step_id}: quality targets met.")


def print_final_results(
    num_steps,
    max_steps,
    runtime_seconds,
    total_cost,
    total_tokens,
    prompt_tokens,
    completion_tokens,
    step_0_metrics,
    final_metrics,
    step_0_score,
    final_score,
    baseline_prompt=None,
    optimized_prompt=None,
    baseline_json=None,
    optimized_json=None,
):
    """Print final optimization results with before/after comparison."""
    print()
    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print()

    # Summary
    stop_reason = "early stopping" if num_steps < max_steps else "max steps"
    print(f"  Steps completed:  {num_steps} / {max_steps} ({stop_reason})")
    print(f"  Runtime:          {runtime_seconds:.1f}s")
    cost_str = "N/A (pricing unavailable for model)" if math.isnan(total_cost) else f"${total_cost:.6f}"
    print(f"  Total cost:       {cost_str}")
    print(f"  Total tokens:     {total_tokens:,} (prompt: {prompt_tokens:,} / completion: {completion_tokens:,})")
    print()

    # Quality comparison table
    if step_0_metrics and final_metrics:
        print("  Quality Improvement (Step 0 -> Final):")
        print(f"  {'Metric':<22} {'Step 0':>10} {'Final':>10} {'Change':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")

        rows = [
            (
                "Schema Valid Rate",
                step_0_metrics.get("schema_valid_rate", 0),
                final_metrics.get("schema_valid_rate", 0),
                True,  # is percentage
            ),
            (
                "Field Overlap",
                step_0_metrics.get("field_overlap_mean", 0),
                final_metrics.get("field_overlap_mean", 0),
                False,
            ),
            (
                "Value Similarity",
                step_0_metrics.get("value_similarity_mean", 0),
                final_metrics.get("value_similarity_mean", 0),
                False,
            ),
            (
                "JSON Distance",
                step_0_metrics.get("json_distance_mean", 0),
                final_metrics.get("json_distance_mean", 0),
                False,
            ),
            ("Composite Score", step_0_score, final_score, False),
        ]

        for name, v0, vf, is_pct in rows:
            change = vf - v0
            sign = "+" if change >= 0 else ""
            if is_pct:
                print(
                    f"  {name:<22} {v0*100:>9.1f}% {vf*100:>9.1f}% {sign}{change*100:>8.1f}%"
                )
            else:
                print(
                    f"  {name:<22} {v0:>10.3f} {vf:>10.3f} {sign}{change:>9.3f}"
                )
        print()

    # Show baseline vs optimized extraction on user's text
    if baseline_json:
        print("  Baseline extraction (default prompt) on your text:")
        _print_json_preview(baseline_json)
        print()
    if optimized_json:
        print("  Optimized extraction on your text:")
        _print_json_preview(optimized_json)
        print()

    # Baseline prompt
    print("=" * 64)
    print("  BASELINE EXTRACTION PROMPT")
    print("=" * 64)
    print()
    if baseline_prompt:
        for line in baseline_prompt.splitlines():
            print(f"  {line}")
    else:
        print("  (No baseline prompt)")
    print()

    # Optimized prompt
    print("=" * 64)
    print("  OPTIMIZED EXTRACTION PROMPT")
    print("=" * 64)
    print()
    if optimized_prompt:
        for line in optimized_prompt.splitlines():
            print(f"  {line}")
    else:
        print("  (No prompt refinement produced — baseline prompt was best)")
    print()
    print("=" * 64)
    print()


def _print_json_preview(json_str, max_lines=12):
    """Print a truncated JSON preview."""
    if not json_str:
        print("    (empty)")
        return
    lines = json_str.splitlines()
    for line in lines[:max_lines]:
        print(f"    {line}")
    if len(lines) > max_lines:
        print(f"    ... ({len(lines) - max_lines} more lines)")
