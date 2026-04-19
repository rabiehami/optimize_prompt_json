
# Suppress Hugging Face Hub warnings and progress bars
import os
os.environ["TRANSFORMERS_NO_TQDM"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

"""Main optimization pipeline for JSON extraction prompt refinement."""

import asyncio
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from deepdiff import DeepDiff
from sqlalchemy import text

from optimize_prompt_json import (
    PROMPT_TYPE_JSON_EXTRACTION,
    PROMPT_TYPE_JSON_GENERATION,
    PROMPT_TYPE_REFINEMENT,
    PROMPT_TYPE_TEXT_GENERATION,
    PROMPT_TYPE_BASELINE_EXTRACTION,
)
from optimize_prompt_json.database import (
    LLMResponse,
    Run,
    RunMetric,
    RunStepMetric,
    get_engine,
    get_session,
    init_db,
)
from optimize_prompt_json.json_utils import (
    build_exclude_paths_from_blacklist,
    extract_json_from_response,
    flatten_json,
    is_field_blacklisted,
)
from optimize_prompt_json.llm import ask_model, parallel_requests
from optimize_prompt_json.metrics import (
    compute_composite_score,
    compute_json_metrics,
    get_field_distance_breakdown,
)
from optimize_prompt_json.prompts import (
    create_prompts_for_article_generation,
    create_prompts_for_rand_json,
    extract_json_from_text,
    generate_lessons_learned,
)
from optimize_prompt_json import console

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """All configuration for an optimization run."""

    schema: dict
    text: str
    schema_path: str = ""
    text_path: str = ""
    llm_model: str = "groq/llama-3.1-8b-instant"
    llm_text_gen_model: str | None = None
    llm_optimizer_model: str | None = None
    batch_size: int = 10
    max_steps: int = 10
    min_steps: int = 0
    temp_json: float = 0.5
    temp_extract: float = 0.0
    temp_article: float = 0.0
    field_overlap_target: float = 0.99
    json_distance_target: float = 0.01
    schema_valid_target: float = 0.99
    rollback_threshold: float = 0.01
    rate_limit_delay: float = 0.0
    api_key: str = ""
    api_key_text_gen: str = ""
    api_key_optimizer: str = ""
    evaluate_only: bool = False
    db_url: str | None = None
    log_dir: str | None = None
    quiet: bool = False
    text_gen_max_tokens: int = 5000
    max_tokens: int = 8192
    blacklist_fields: set = field(default_factory=lambda: {
        "id",
        "uuid",
        "_id",
        "object_id",
        "timestamp",
        "created_at",
        "updated_at",
    })
    initial_prompt: str = (
        "Please extract from the text below the data described in the schema below as a JSON object."
    )


# =====================================================================
# Helper Functions
# =====================================================================


def _extract_required_field_names(schema):
    """Recursively collect every field name listed under 'required' in a schema."""
    names = set()
    if not isinstance(schema, dict):
        return names
    for field_name in schema.get("required", []):
        names.add(field_name)
    for prop_schema in schema.get("properties", {}).values():
        names |= _extract_required_field_names(prop_schema)
    for def_schema in schema.get("$defs", {}).values():
        names |= _extract_required_field_names(def_schema)
    return names


def _load_step_df(run_id, step_id):
    """Load all LLM responses for a given step."""
    return pd.read_sql(
        text("SELECT * FROM llm_responses WHERE run_id = :r AND step_id = :s"),
        get_engine(),
        params={"r": run_id, "s": step_id},
    )


def _extract_json_for_step(run_id, step_id):
    """Parse JSON from extraction responses and update DB."""
    session = get_session()
    rows = (
        session.query(LLMResponse)
        .filter(LLMResponse.run_id == run_id)
        .filter(LLMResponse.step_id == step_id)
        .filter(LLMResponse.prompt_type == PROMPT_TYPE_JSON_EXTRACTION)
        .all()
    )
    for r in rows:
        try:
            r.json = json.dumps(extract_json_from_response(r.content), indent=2)
        except Exception:
            r.json = None
    session.commit()
    session.close()


def _should_stop(step_metrics, step_id, config):
    """Check stopping criteria."""
    if step_id >= config.max_steps - 1:
        return True
    if step_id < config.min_steps:
        return False
    if not step_metrics:
        return False
    if step_metrics["schema_valid_rate"] < config.schema_valid_target:
        return False
    if (
        step_metrics["field_overlap_mean"] >= config.field_overlap_target
        and step_metrics["json_distance_mean"] <= config.json_distance_target
    ):
        return True
    return False


def _get_field_distance_breakdown_for_refinement(run_id, step_id, schema):
    """Aggregate field distances (differences only) for prompt refinement."""
    df = _load_step_df(run_id, step_id)
    orig = df[df.prompt_type == PROMPT_TYPE_JSON_GENERATION]
    extr = df[df.prompt_type == PROMPT_TYPE_JSON_EXTRACTION]
    merged = orig.merge(extr, on="artifact_id", suffixes=("_orig", "_extr"))

    aggregated = {}
    for _, row in merged.iterrows():
        try:
            original = json.loads(row.json_orig)
            extracted = json.loads(row.json_extr)
            field_distances = get_field_distance_breakdown(original, extracted, schema)
            for fp, bd in field_distances.items():
                if bd.get("distance", 0) > 0:
                    aggregated.setdefault(fp, []).append(bd)
        except Exception:
            continue

    result = {}
    for fp, breakdowns in aggregated.items():
        avg_dist = np.mean([b["distance"] for b in breakdowns])
        best = max(breakdowns, key=lambda b: b["distance"])
        best["distance"] = round(avg_dist, 4)
        result[fp] = best

    return dict(sorted(result.items(), key=lambda x: x[1].get("distance", 0), reverse=True))


def _get_step_diffs(run_id, step_id, schema, blacklist_fields=None):
    """Calculate DeepDiff between original and extracted JSON per artifact."""

    def _lower_json(obj):
        if isinstance(obj, dict):
            return {str(k).lower(): _lower_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_lower_json(i) for i in obj]
        elif isinstance(obj, str):
            return obj.lower()
        return obj

    def _make_serializable(obj):
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    df = _load_step_df(run_id, step_id)
    orig = df[df.prompt_type == PROMPT_TYPE_JSON_GENERATION]
    extr = df[df.prompt_type == PROMPT_TYPE_JSON_EXTRACTION]
    merged = orig.merge(extr, on="artifact_id", suffixes=("_orig", "_extr"))
    diffs = []

    for _, row in merged.iterrows():
        try:
            original = json.loads(row.json_orig)
            extracted = json.loads(row.json_extr)
            exclude_paths = build_exclude_paths_from_blacklist(
                _lower_json(original),
                field_blacklist={f.lower() for f in blacklist_fields} if blacklist_fields else None,
            )
            diff = DeepDiff(
                _lower_json(original), _lower_json(extracted),
                ignore_order=True, exclude_paths=exclude_paths,
            )
            diff_dict = diff.to_dict() if hasattr(diff, "to_dict") else dict(diff)
            diff_dict = {k: _make_serializable(v) for k, v in diff_dict.items()}
            diffs.append({"artifact_id": row.artifact_id, "diff": diff_dict})
        except Exception:
            continue
    return diffs


# =====================================================================
# Evaluation
# =====================================================================


def _evaluate_step(run_id, step_id, schema):
    """Evaluate quality metrics for all artifacts in a step."""
    df = _load_step_df(run_id, step_id)
    orig = df[df.prompt_type == PROMPT_TYPE_JSON_GENERATION]
    extr = df[df.prompt_type == PROMPT_TYPE_JSON_EXTRACTION]
    merged = orig.merge(extr, on="artifact_id", suffixes=("_orig", "_extr"))

    if merged.empty:
        logger.warning(f"Step {step_id}: merged DataFrame empty.")
        return

    session = get_session()
    cumulative = 0.0

    for _, row in merged.iterrows():
        try:
            original = json.loads(row.json_orig)
            extracted = json.loads(row.json_extr)
        except Exception:
            continue

        metrics = compute_json_metrics(original, extracted, schema)
        cumulative += row["price_orig"]

        session.add(
            RunMetric(
                run_id=run_id,
                artifact_id=row["artifact_id"],
                step_id=step_id,
                price_step=row["price_orig"],
                price_cumulative=cumulative,
                **metrics,
            )
        )
    session.commit()
    session.close()


def _aggregate_step_metrics(run_id, step_id):
    """Aggregate and store step-level metrics."""
    df = pd.read_sql(
        text("SELECT * FROM run_metrics WHERE run_id = :run AND step_id <= :step"),
        get_engine(),
        params={"run": run_id, "step": step_id},
    )
    if df.empty:
        return None

    df_current = df[df["step_id"] == step_id]
    agg = {
        "run_id": run_id,
        "step_id": step_id,
        "batch_size": len(df_current),
        "price_step": df_current["price_step"].sum(),
        "price_cumulative": df["price_step"].sum(),
        "field_overlap_mean": df_current["field_overlap"].mean(),
        "field_overlap_std": df_current["field_overlap"].std(),
        "value_similarity_mean": df_current["value_similarity"].mean(),
        "value_similarity_std": df_current["value_similarity"].std(),
        "json_distance_mean": df_current["json_distance"].mean(),
        "json_distance_std": df_current["json_distance"].std(),
        "schema_valid_rate": df_current["schema_valid"].mean(),
    }

    session = get_session()
    session.add(RunStepMetric(**agg))
    session.commit()
    session.close()
    return agg


# =====================================================================
# Refinement
# =====================================================================


async def _refine_extraction_prompt(
    run_id, step_id, prev_prompt, json_schema, lessons_section, config,
):
    """Ask the LLM to refine the extraction prompt."""
    llm_model = config.llm_optimizer_model or config.llm_model
    prompt = f"""
You are an expert at designing prompts for extracting JSON from text using LLMs.

Below is the previous extraction prompt:
---
{prev_prompt}
---
{lessons_section}

Your task:
1. Aggregate the previous prompt with the lessons learned above into an improved extraction prompt.

CRITICAL CONSTRAINTS:
- Do NOT simply append new rules. Merge lessons into existing instructions where they overlap.
- If a lesson is already covered by the previous prompt (even with different wording), do NOT add it again.
- Only make the refined prompt longer if a genuinely new insight requires it.
- Prefer concise, direct instructions over verbose explanations. Remove redundant phrasing.
- The schema is RIGID and will not be changed. Focus entirely on improving the extraction prompt.

Return your answer as a JSON object with one field:
'refined_prompt': <your improved extraction prompt>
"""
    meta = [
        dict(
            run_id=run_id,
            request_id=str(uuid4()),
            parent_request_id=None,
            artifact_id=None,
            group_id=None,
            step_id=step_id,
            llm_model=llm_model,
            llm_temperature=0.2,
            json_schema=json.dumps(json_schema),
            prompt_type=PROMPT_TYPE_REFINEMENT,
            api_key=config.api_key_optimizer or config.api_key,
            max_tokens=config.max_tokens,
        )
    ]
    results = await parallel_requests([prompt], meta, rate_limit_delay=config.rate_limit_delay)
    try:
        result = extract_json_from_response(results[0]["content"])
        return result.get("refined_prompt", prev_prompt)
    except Exception as e:
        logger.warning(f"Failed to extract refined prompt: {e}. Using previous.")
        return prev_prompt


async def _summarize_and_deduplicate_lessons(accumulated, current, config):
    """Combine and deduplicate lessons from previous and current steps."""
    llm_model = config.llm_optimizer_model or config.llm_model
    prompt = f"""You are an expert at analyzing extraction patterns and quality issues.

I have lessons learned from multiple optimization steps. Your task is to combine and deduplicate them:

ACCUMULATED LESSONS FROM PREVIOUS STEPS:
---
{accumulated}
---

CURRENT STEP LESSONS LEARNED:
---
{current}
---

Your task:
1. Identify common themes between accumulated and current lessons
2. Merge duplicate or similar lessons into one concise statement
3. Keep lessons organized by problem category
4. Remove redundant wording while preserving all unique insights
5. Maintain concrete examples where helpful

Return your answer as a plain text list of deduplicated lessons (no JSON), one lesson per line, starting with a dash (-).
Do not include explanations or preamble, just the deduplicated lessons list.
"""
    meta = [
        dict(
            run_id="summarize-lessons",
            request_id=str(uuid4()),
            parent_request_id=None,
            artifact_id=None,
            group_id=None,
            step_id=None,
            llm_model=llm_model,
            llm_temperature=0.2,
            json_schema="",
            prompt_type="lesson_summarization",
            api_key=config.api_key_optimizer or config.api_key,
            max_tokens=config.max_tokens,
        )
    ]
    try:
        results = await parallel_requests([prompt], meta, rate_limit_delay=config.rate_limit_delay)
        return results[0].get("content", current).strip()
    except Exception as e:
        logger.warning(f"Failed to summarize lessons: {e}. Using current.")
        return current


# =====================================================================
# Single Optimization Step
# =====================================================================


async def _run_step(config, run_id, step_id, prev_extract_prompt=None, accumulated_lessons=None):
    """Execute one optimization step: generate → text → extract → evaluate → refine."""
    schema = config.schema
    extract_prompt_template = prev_extract_prompt
    artifact_ids = [str(uuid4()) for _ in range(config.batch_size)]

    # --- Phase 1: Random JSON generation ---
    rand_group_id = str(uuid4())
    rand_prompts = create_prompts_for_rand_json(schema, config.batch_size, max_tokens=config.text_gen_max_tokens)
    rand_meta = [
        dict(
            run_id=run_id,
            request_id=str(uuid4()),
            parent_request_id=None,
            artifact_id=aid,
            group_id=rand_group_id,
            step_id=step_id,
            llm_model=config.llm_model,
            llm_temperature=config.temp_json,
            json_schema=json.dumps(schema),
            prompt_type=PROMPT_TYPE_JSON_GENERATION,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
        )
        for aid in artifact_ids
    ]
    rand = await parallel_requests(rand_prompts, rand_meta, rate_limit_delay=config.rate_limit_delay)

    # Validate generated JSON against schema structure
    schema_props = set(schema.get("properties", {}).keys())
    if schema_props:
        valid_rand = []
        for r in rand:
            parsed = (
                extract_json_from_response(r["content"])
                if r.get("json") is None
                else json.loads(r["json"])
            )
            if not isinstance(parsed, dict):
                logger.warning(f"Discarding artifact {r.get('artifact_id')}: not a dict.")
                continue
            overlap = len(set(parsed.keys()) & schema_props) / max(len(parsed.keys()), 1)
            if overlap < 0.4:
                logger.warning(
                    f"Discarding artifact {r.get('artifact_id')}: "
                    f"root key overlap {overlap:.0%} < 40%."
                )
                continue
            valid_rand.append(r)
        if len(valid_rand) < len(rand):
            logger.info(
                f"Ground-truth validation: discarded {len(rand) - len(valid_rand)} artifact(s)."
            )
        rand = valid_rand

    if not rand:
        logger.error("All generated artifacts were invalid. Skipping this step.")
        return None, {"refined_prompt": prev_extract_prompt, "lessons_learned": accumulated_lessons}, schema

    # --- Phase 2: Synthetic text generation ---
    text_gen_model = config.llm_text_gen_model or config.llm_model
    synth_group_id = str(uuid4())
    synth_prompts = create_prompts_for_article_generation(
        [r["content"] for r in rand], reference_text=config.text, max_tokens=config.text_gen_max_tokens
    )
    synth_meta = [
        dict(
            run_id=run_id,
            request_id=str(uuid4()),
            parent_request_id=r["request_id"],
            artifact_id=r["artifact_id"],
            group_id=synth_group_id,
            step_id=step_id,
            llm_model=text_gen_model,
            llm_temperature=config.temp_article,
            json_schema=json.dumps(schema),
            prompt_type=PROMPT_TYPE_TEXT_GENERATION,
            api_key=config.api_key_text_gen or config.api_key,
            max_tokens=config.max_tokens,
        )
        for r in rand
    ]
    synth = await parallel_requests(synth_prompts, synth_meta, rate_limit_delay=config.rate_limit_delay)

    # --- Phase 3: JSON extraction ---
    if extract_prompt_template is None:
        extract_prompts = extract_json_from_text(
            [r["content"] for r in synth], schema,
            refined_prompt=config.initial_prompt,
            accumulated_lessons=accumulated_lessons,
        )
        extract_prompt_template = extract_prompts[0] if extract_prompts else ""
    else:
        if isinstance(extract_prompt_template, dict) and "refined_prompt" in extract_prompt_template:
            refined_prompt = extract_prompt_template["refined_prompt"]
            validation_errors = extract_prompt_template.get("validation_errors")
            diffs = extract_prompt_template.get("diffs")
            fdb = extract_prompt_template.get("field_distance_breakdown")
        else:
            refined_prompt = extract_prompt_template
            validation_errors = diffs = fdb = None

        extract_prompts = extract_json_from_text(
            [r["content"] for r in synth], schema,
            refined_prompt=refined_prompt,
            validation_errors=validation_errors,
            diffs=diffs,
            field_distance_breakdown=fdb,
            accumulated_lessons=accumulated_lessons,
        )

    extract_group_id = str(uuid4())
    extract_meta = [
        dict(
            run_id=run_id,
            request_id=str(uuid4()),
            parent_request_id=r["request_id"],
            artifact_id=r["artifact_id"],
            group_id=extract_group_id,
            step_id=step_id,
            llm_model=config.llm_model,
            llm_temperature=config.temp_extract,
            json_schema=json.dumps(schema),
            prompt_type=PROMPT_TYPE_JSON_EXTRACTION,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
        )
        for r in synth
    ]
    await parallel_requests(extract_prompts, extract_meta, rate_limit_delay=config.rate_limit_delay)

    # --- Phase 4: Extract JSON fields ---
    _extract_json_for_step(run_id, step_id)

    # --- Phase 5: Evaluate & aggregate ---
    _evaluate_step(run_id, step_id, schema)
    step_metrics = _aggregate_step_metrics(run_id, step_id)

    # --- Phase 6: Refine prompt ---
    if not config.evaluate_only:
        fdb = _get_field_distance_breakdown_for_refinement(run_id, step_id, schema)
        diffs_list = _get_step_diffs(run_id, step_id, schema, blacklist_fields=config.blacklist_fields)
        diffs_for_feedback = [d["diff"] for d in diffs_list if d.get("diff")]

        current_lessons = generate_lessons_learned(fdb, schema=schema)

        if accumulated_lessons and current_lessons:
            combined_lessons = await _summarize_and_deduplicate_lessons(
                accumulated_lessons, current_lessons, config
            )
        elif accumulated_lessons:
            combined_lessons = accumulated_lessons
        elif current_lessons:
            combined_lessons = current_lessons
        else:
            combined_lessons = None

        lessons_section = f"\n{combined_lessons}" if combined_lessons else ""

        refined_prompt = await _refine_extraction_prompt(
            run_id, step_id, extract_prompt_template, schema, lessons_section, config,
        )

        refined_prompt_with_feedback = {
            "refined_prompt": refined_prompt,
            "lessons_learned": combined_lessons,
            "validation_errors": None,
            "diffs": diffs_for_feedback,
            "field_distance_breakdown": fdb,
        }
    else:
        refined_prompt_with_feedback = {
            "refined_prompt": None,
            "lessons_learned": None,
            "validation_errors": None,
            "diffs": None,
            "field_distance_breakdown": None,
        }

    return step_metrics, refined_prompt_with_feedback, schema


# =====================================================================
# Baseline / Final Extraction on User Text
# =====================================================================


async def _extract_from_user_text(config, run_id, prompt_override=None, step_id=-1):
    """Apply an extraction prompt to the user's actual text and return (json_str, prompt_used)."""

    schema = config.schema
    # Use prompt_override if provided, else use config.initial_prompt
    refined_prompt = prompt_override if prompt_override is not None else config.initial_prompt
    prompts = extract_json_from_text([config.text], schema, refined_prompt=refined_prompt)

    if not prompts:
        return None, None

    prompt_used = prompts[0]

    meta = dict(
        run_id=run_id,
        request_id=str(uuid4()),
        parent_request_id=None,
        artifact_id="user_text",
        group_id=str(uuid4()),
        step_id=step_id,
        llm_model=config.llm_model,
        llm_temperature=config.temp_extract,
        json_schema=json.dumps(schema),
        prompt_type=PROMPT_TYPE_BASELINE_EXTRACTION,
        api_key=config.api_key,
        max_tokens=config.max_tokens,
    )
    response = await ask_model(
        prompt=prompts[0], rate_limit_delay=config.rate_limit_delay, **meta
    )

    try:
        parsed = extract_json_from_response(response["content"])
        response["json"] = json.dumps(parsed, indent=2)
    except Exception:
        response["json"] = None

    session = get_session()
    session.add(LLMResponse(**response))
    session.commit()
    session.close()

    return response.get("json"), prompt_used
# =====================================================================


async def run_optimization(config: OptimizationConfig):
    """Run the full optimization pipeline and return the optimized prompt."""
    start_time = time.time()

    # Initialize database (None -> in-memory SQLite, no file written)
    db_url = config.db_url if config.db_url is not None else "sqlite:///:memory:"
    init_db(db_url)

    run_id = str(uuid4())
    run_created = datetime.now()
    date_str = run_created.strftime("%y%m%d%H%M")
    short_id = run_id[:8]

    # Setup logging
    if config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        log_file = os.path.join(config.log_dir, f"run_{date_str}_{short_id}.log")
        logging.basicConfig(
            filename=log_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(funcName)s: %(message)s",
            level=logging.INFO,
        )

    # Print header
    if not config.quiet:
        console.print_header(config)

    # Create run record
    session = get_session()
    text_gen_model = config.llm_text_gen_model or config.llm_model
    optimizer_model = config.llm_optimizer_model or config.llm_model
    session.add(
        Run(
            run_id=run_id,
            llm_model=config.llm_model,
            llm_text_gen_model=text_gen_model,
            llm_optimizer_model=optimizer_model,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            llm_temp_json_generation=config.temp_json,
            llm_temp_text_generation=config.temp_article,
            llm_temp_json_extraction=config.temp_extract,
            json_schema=json.dumps(config.schema),
            created=run_created,
        )
    )
    session.commit()
    session.close()

    # Baseline extraction on user's text
    baseline_json, baseline_prompt = await _extract_from_user_text(config, run_id, prompt_override=None, step_id=-1)
    logger.info(f"Baseline extraction: {baseline_json}")

    # --- Optimization loop ---
    if config.evaluate_only:
        config.max_steps = 1

    refined_prompt = None
    accumulated_lessons = None
    num_steps = 0
    best_prompt = None
    best_score = -1.0
    prev_score = -1.0
    step_0_metrics = None
    step_0_score = 0.0
    final_metrics = None
    final_score = 0.0

    for step_id in range(config.max_steps):
        current_prompt = refined_prompt

        result = await _run_step(
            config, run_id, step_id,
            prev_extract_prompt=refined_prompt,
            accumulated_lessons=accumulated_lessons,
        )

        step_metrics, refined_prompt_with_feedback, _ = result
        current_score = compute_composite_score(step_metrics)
        new_refined_prompt = refined_prompt_with_feedback.get("refined_prompt")

        if step_id == 0:
            step_0_metrics = step_metrics
            step_0_score = current_score

        # --- Rollback logic ---
        rolled_back = False
        schema_valid_rate = (
            step_metrics.get("schema_valid_rate", 1.0) if step_metrics else 1.0
        )

        if schema_valid_rate == 0.0:
            logger.warning(
                f"Step {step_id}: schema_valid_rate=0 — rolling back to best prompt."
            )
            refined_prompt = best_prompt
            rolled_back = True
        elif prev_score >= 0 and current_score < prev_score - config.rollback_threshold:
            logger.warning(
                f"Step {step_id}: score dropped {prev_score:.4f} -> {current_score:.4f}. Rolling back."
            )
            refined_prompt = best_prompt
            accumulated_lessons = refined_prompt_with_feedback.get("lessons_learned")
            rolled_back = True
        else:
            accumulated_lessons = refined_prompt_with_feedback.get("lessons_learned")
            refined_prompt = new_refined_prompt
            if current_score > best_score:
                best_score = current_score
                best_prompt = current_prompt
                logger.info(f"Step {step_id}: new best (score={best_score:.4f})")

        # Console output
        if not config.quiet:
            console.print_step(
                step_id, config.max_steps, current_score, prev_score, step_metrics,
                rolled_back=rolled_back,
            )

        prev_score = current_score
        final_metrics = step_metrics
        final_score = current_score
        num_steps = step_id + 1

        if _should_stop(step_metrics, step_id, config):
            if not config.quiet:
                console.print_early_stop(step_id)
            break

    # --- Final extraction with optimized prompt ---
    optimized_prompt_text = best_prompt or refined_prompt
    optimized_json = None
    if optimized_prompt_text and config.text:
        optimized_json, _ = await _extract_from_user_text(
            config, run_id, prompt_override=optimized_prompt_text, step_id=num_steps,
        )

    # --- Calculate totals ---
    df_responses = pd.read_sql(
        text("SELECT prompt_tokens, completion_tokens, price FROM llm_responses WHERE run_id = :r"),
        get_engine(),
        params={"r": run_id},
    )
    total_prompt_tokens = int(df_responses["prompt_tokens"].sum()) if not df_responses.empty else 0
    total_completion_tokens = int(df_responses["completion_tokens"].sum()) if not df_responses.empty else 0
    if not df_responses.empty:
        total_price = float(df_responses["price"].fillna(0).sum())
    else:
        total_price = 0.0
    total_runtime = time.time() - start_time

    # Update run record
    session = get_session()
    run = session.query(Run).filter(Run.run_id == run_id).first()
    if run:
        run.num_steps = num_steps
        run.total_prompt_tokens = total_prompt_tokens
        run.total_completion_tokens = total_completion_tokens
        run.total_price = total_price
        run.total_runtime_seconds = total_runtime
        if final_metrics:
            run.final_field_overlap = final_metrics.get("field_overlap_mean")
            run.final_value_similarity = final_metrics.get("value_similarity_mean")
            run.final_json_distance = final_metrics.get("json_distance_mean")
            run.final_schema_valid_rate = final_metrics.get("schema_valid_rate")
        session.commit()
    session.close()

    # Print final results
    if not config.quiet:
        console.print_final_results(
            num_steps=num_steps,
            max_steps=config.max_steps,
            runtime_seconds=total_runtime,
            total_cost=total_price,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            step_0_metrics=step_0_metrics,
            final_metrics=final_metrics,
            step_0_score=step_0_score,
            final_score=final_score,
            baseline_prompt=baseline_prompt,
            optimized_prompt=optimized_prompt_text,
            baseline_json=baseline_json,
            optimized_json=optimized_json,
        )

    cost_str = "N/A" if math.isnan(total_price) else f"${total_price:.6f}"
    logger.info(f"Done. Run ID: {run_id}, Steps: {num_steps}, Cost: {cost_str}")

    return {
        "run_id": run_id,
        "optimized_prompt": optimized_prompt_text,
        "num_steps": num_steps,
        "final_score": final_score,
        "step_0_score": step_0_score,
        "baseline_json": baseline_json,
        "optimized_json": optimized_json,
        "total_cost": total_price,
        "total_runtime": total_runtime,
    }
