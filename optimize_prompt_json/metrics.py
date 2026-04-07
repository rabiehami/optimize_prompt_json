"""Similarity and distance metrics for JSON comparison."""

import difflib
import logging
import re

import numpy as np
from deepdiff import DeepDiff
from jsonschema import ValidationError, validate
from scipy.spatial.distance import cosine

from optimize_prompt_json.json_utils import (
    build_exclude_paths_from_blacklist,
    extract_arrays_from_json,
    flatten_json,
    is_field_blacklisted,
)

logger = logging.getLogger(__name__)

LONG_FORM_WORD_THRESHOLD = 5
NUMERIC_EPSILON = 1e-6

_embedding_model = None


def get_embedding_model():
    """Lazy-load sentence-transformers embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            _embedding_model.encode("warmup", show_progress_bar=False)
            logger.info("Embedding model loaded: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            _embedding_model = None
    return _embedding_model


def get_field_type_from_schema(field_path, schema):
    """Extract the JSON schema type for a given field path."""
    if not schema or "properties" not in schema:
        return "unknown"
    top_field = field_path.split(".")[0].split("[")[0]
    properties = schema.get("properties", {})
    if top_field not in properties:
        return "unknown"
    field_schema = properties[top_field]
    if "enum" in field_schema:
        return "enum"
    return field_schema.get("type", "string")


def compute_numeric_distance(val1, val2):
    """Normalized numeric distance: |x-y| / max(|x|,|y|,eps)."""
    try:
        x, y = float(val1), float(val2)
        return min(abs(x - y) / max(abs(x), abs(y), NUMERIC_EPSILON), 1.0)
    except (ValueError, TypeError):
        return 1.0


def compute_levenshtein_distance(str1, str2):
    """Normalized Levenshtein distance via SequenceMatcher."""
    s1, s2 = str(str1).lower(), str(str2).lower()
    if s1 == s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_embedding_distance(str1, str2):
    """Cosine distance between sentence embeddings. Falls back to Levenshtein."""
    model = get_embedding_model()
    if model is None:
        return compute_levenshtein_distance(str1, str2)
    try:
        emb1 = model.encode(str(str1), convert_to_numpy=True, show_progress_bar=False)
        emb2 = model.encode(str(str2), convert_to_numpy=True, show_progress_bar=False)
        return min(float(cosine(emb1, emb2)), 1.0)
    except Exception as e:
        logger.warning(f"Embedding computation failed: {e}. Falling back to Levenshtein.")
        return compute_levenshtein_distance(str1, str2)


def is_long_form_text(value):
    """Heuristic: True if >= LONG_FORM_WORD_THRESHOLD words."""
    return len(str(value).strip().split()) >= LONG_FORM_WORD_THRESHOLD


def compute_weighted_value_similarity(original, extracted, schema):
    """Schema-aware weighted value similarity using hybrid distance metrics."""
    flat_orig = flatten_json(original)
    flat_extr = flatten_json(extracted)
    field_distances = []

    for path, orig_value in flat_orig:
        path_lower = path.lower()
        field_name = path.split("[")[-1].strip("']\"") if "[" in path else path
        if is_field_blacklisted(field_name):
            continue

        extr_value = None
        for extr_path, extr_val in flat_extr:
            if extr_path.lower() == path_lower:
                extr_value = extr_val
                break

        if extr_value is None:
            field_distances.append(1.0)
            continue

        field_type = get_field_type_from_schema(path, schema)
        if field_type in ("number", "integer"):
            distance = compute_numeric_distance(orig_value, extr_value)
        elif field_type in ("boolean", "enum"):
            distance = 0.0 if orig_value.lower() == extr_value.lower() else 1.0
        elif is_long_form_text(orig_value):
            distance = compute_embedding_distance(orig_value, extr_value)
        else:
            distance = compute_levenshtein_distance(orig_value, extr_value)
        field_distances.append(distance)

    if not field_distances:
        return 1.0
    return float(1.0 - np.mean(field_distances))


def compute_array_optimal_matching(original_array, extracted_array):
    """Greedy best-match alignment between two arrays of strings."""
    if not original_array:
        return {}
    distances = {}
    for i, orig_val in enumerate(original_array):
        best_idx, best_dist = None, 1.0
        for j, extr_val in enumerate(extracted_array):
            if is_long_form_text(orig_val):
                d = compute_embedding_distance(orig_val, extr_val)
            else:
                d = compute_levenshtein_distance(orig_val, extr_val)
            if d < best_dist:
                best_dist = d
                best_idx = j
        distances[i] = (best_idx, best_dist)
    return distances


def get_field_distance_breakdown(original, extracted, schema):
    """Per-field distance breakdown with optimal array element matching."""
    orig_arrays = extract_arrays_from_json(original)
    extr_arrays = extract_arrays_from_json(extracted)
    array_used_indices = {}
    flat_orig = flatten_json(original)
    flat_extr = flatten_json(extracted)
    field_breakdown = {}

    for path, orig_value in flat_orig:
        path_lower = path.lower()
        field_name = path.split("[")[-1].strip("']\"") if "[" in path else path
        if is_field_blacklisted(field_name):
            continue

        is_array_element = "[" in path
        array_path = array_index = None
        if is_array_element:
            match = re.match(r"(.*?)\[(\d+)\]$", path)
            if match:
                array_path = match.group(1)
                array_index = int(match.group(2))

        extr_value = None
        if is_array_element and array_path in orig_arrays and array_path in extr_arrays:
            if array_path not in array_used_indices:
                array_used_indices[array_path] = set()
            matches = compute_array_optimal_matching(
                orig_arrays[array_path], extr_arrays[array_path]
            )
            if array_index in matches:
                best_idx, _ = matches[array_index]
                if best_idx is not None:
                    extr_value = extr_arrays[array_path][best_idx]
        else:
            for extr_path, extr_val in flat_extr:
                if extr_path.lower() == path_lower:
                    extr_value = extr_val
                    break

        if extr_value is None:
            field_breakdown[path] = {
                "original": str(orig_value)[:100],
                "extracted": None,
                "distance": 1.0,
                "metric_type": "missing_in_extracted",
            }
            continue

        field_type = get_field_type_from_schema(path, schema)
        if field_type in ("number", "integer"):
            distance = compute_numeric_distance(orig_value, extr_value)
            metric_type = "numeric"
        elif field_type in ("boolean", "enum"):
            distance = 0.0 if orig_value.lower() == extr_value.lower() else 1.0
            metric_type = "exact_match"
        elif is_long_form_text(orig_value):
            distance = compute_embedding_distance(orig_value, extr_value)
            metric_type = "embedding"
        else:
            distance = compute_levenshtein_distance(orig_value, extr_value)
            metric_type = "levenshtein"

        field_breakdown[path] = {
            "original": str(orig_value)[:100],
            "extracted": str(extr_value)[:100],
            "distance": round(float(distance), 4),
            "metric_type": metric_type,
        }
    return field_breakdown


def compute_json_metrics(original, extracted, schema):
    """Compute quality metrics comparing original and extracted JSON."""
    flat_a = {(p.lower(), v.lower()) for p, v in flatten_json(original)}
    flat_b = {(p.lower(), v.lower()) for p, v in flatten_json(extracted)}

    def _filter_blacklist(flat_set):
        filtered = set()
        for path, value in flat_set:
            field_name = path.split("[")[-1].strip("']\"") if "[" in path else path
            if not is_field_blacklisted(field_name):
                filtered.add((path, value))
        return filtered

    flat_a = _filter_blacklist(flat_a)
    flat_b = _filter_blacklist(flat_b)

    paths_a = {p for p, _ in flat_a}
    paths_b = {p for p, _ in flat_b}
    field_overlap = (
        len(paths_a & paths_b) / len(paths_a | paths_b)
        if paths_a or paths_b
        else 1.0
    )

    value_similarity = compute_weighted_value_similarity(original, extracted, schema)

    try:
        validate(extracted, schema)
        schema_valid = 1
    except ValidationError:
        schema_valid = 0

    exclude_paths = build_exclude_paths_from_blacklist(original)
    diff = DeepDiff(
        original, extracted, ignore_order=True, ignore_string_case=True,
        exclude_paths=exclude_paths,
    )
    individual_diffs = sum(len(v) for v in diff.values())
    total_paths = max(len(paths_a | paths_b), 1)
    json_distance = min(1.0, individual_diffs / total_paths)

    return {
        "schema_valid": schema_valid,
        "field_overlap": field_overlap,
        "value_similarity": value_similarity,
        "json_distance": json_distance,
    }


def compute_composite_score(metrics):
    """Single quality score (higher = better). Weights: valid 0.4, overlap 0.2, similarity 0.2, 1-distance 0.2."""
    if not metrics:
        return 0.0
    return (
        metrics.get("schema_valid_rate", 0.0) * 0.40
        + metrics.get("field_overlap_mean", 0.0) * 0.20
        + metrics.get("value_similarity_mean", 0.0) * 0.20
        + (1.0 - metrics.get("json_distance_mean", 1.0)) * 0.20
    )
