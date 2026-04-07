"""Prompt generation for JSON creation, text generation, and extraction."""

import json
import logging
import re



logger = logging.getLogger(__name__)


DEFAULT_MAX_TOKENS = 1000


def create_prompts_for_rand_json(json_schema, batch_size=1, max_tokens=None):
    """Generate prompts that instruct an LLM to create random JSON examples from a schema."""
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    meta_prompt = (
        "Please create a random, realistic JSON example based on the schema below. "
        "Be creative and original — invent a plausible, varied scenario. "
        "Rules for field values:\n"
        "  - String fields: complete, realistic phrases or sentences of at least 3 words. "
        "Never use single characters, abbreviations alone, or placeholder text such as 'N/A', 'string', or 'value'.\n"
        "  - Integer / number fields: realistic non-trivial values (not just 0 or 1).\n"
        "  - Array fields: include at least 2 distinct, meaningful entries.\n"
        "Output ONLY a JSON object whose top-level keys exactly match the schema's top-level properties. "
        "Do NOT wrap the output in any other key, do NOT include any JSON Schema meta-fields such as "
        "'$schema', '$id', '$defs', or '$ref'. "
        "Do NOT use markdown or code blocks. The output must be valid JSON and directly parsable. "
        "Please do not exceed {max_token} tokens with your answer.\n"
        "Here the schema: {json_schema}"
    )
    return [
        meta_prompt.format(max_token=max_tokens, json_schema=json.dumps(json_schema))
        for _ in range(batch_size)
    ]


def create_prompts_for_article_generation(json_data_list, reference_text=None, max_tokens=None):
    """Generate prompts that convert JSON data into natural-language text."""
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    base_prompt = (
        "Please write a text that is based on the json data below. "
        "CRITICAL REQUIREMENT: Every single value from the JSON must appear in your text — including all "
        "numeric codes, zip codes, postal codes, dates, identifiers, and any other structured or coded values. "
        "These must be reproduced EXACTLY and VERBATIM as they appear in the JSON (e.g. if the JSON has "
        '"zip": "94105", the text must contain the string \'94105\' literally). '
        "Do NOT paraphrase, omit, or approximate any such values. "
        "The data can be embedded in natural human-language phrasing, but the values themselves must not be altered. "
        "You can add side information unrelated to the schema. "
        "The text should sound like a normal/typical text and should be unstructured. "
        "Please output solely the article without explanations etc. of what you did. Do NOT use markdown or code blocks. "
        "Please do not exceed {max_token} tokens with your answer."
    )
    if reference_text:
        base_prompt += (
            "\n\nThe text you write should be similar in style, format, and structure to the following example text "
            "(but with completely different data based on the json provided):\n"
            '"""\n{reference_text}\n"""'
        )
    meta_prompt = base_prompt + "\n\nHere the json data: {json_data}"

    prompts = []
    for json_data in json_data_list:
        prompts.append(
            meta_prompt.format(
                max_token=max_tokens,
                reference_text=reference_text or "",
                json_data=json.dumps(json_data),
            )
        )
    return prompts


def extract_json_from_text(
    texts,
    json_schema,
    refined_prompt=None,
    validation_errors=None,
    diffs=None,
    field_distance_breakdown=None,
    accumulated_lessons=None,
):
    """Generate prompts that instruct an LLM to extract JSON from natural-language text."""
    meta_prompt = (
        "{refined_prompt}\n\n"
        "{accumulated_lessons_section}"
        "Text to extract from:\n"
        '"""\n{text}\n"""\n\n'
        "Schema (for reference):\n{json_schema}{mistakes_section}"
    )

    if refined_prompt is None or not isinstance(refined_prompt, str):
        refined_prompt = (
            "From the provided text, extract data and output exactly a JSON object that matches the given schema. "
            "Do not add any extra text or fields. If any field cannot be determined, provide an empty string for that field. "
            "Field names in the output JSON must exactly match the schema field names, "
            "preserving camelCase capitalization (for example, use 'dataCollectionPeriod' not 'datacollectionperiod', "
            "'studyDesign' not 'studydesign', 'outcomeMeasures' not 'outcomemeasures'). "
            "The JSON must be valid and parseable."
        )
    else:
        if "Text to extract from:" in refined_prompt:
            refined_prompt = refined_prompt.split("Text to extract from:")[0].strip()
        if "\nNotes:\n" in refined_prompt:
            refined_prompt = refined_prompt.split("\nNotes:\n")[0].strip()

    accumulated_lessons_section = ""
    if accumulated_lessons:
        accumulated_lessons_section = (
            f"Lessons Learned from Previous Steps (apply these insights):\n{accumulated_lessons}\n\n"
        )

    mistakes_section = ""
    if validation_errors or diffs or field_distance_breakdown:
        mistakes_section = _build_mistakes_section(
            validation_errors, diffs, field_distance_breakdown
        )

    prompts = []
    for text in texts:
        prompts.append(
            meta_prompt.format(
                refined_prompt=refined_prompt.strip(),
                accumulated_lessons_section=accumulated_lessons_section,
                text=text.strip(),
                json_schema=json.dumps(json_schema, indent=2),
                mistakes_section=mistakes_section,
            )
        )
    return prompts


def _build_mistakes_section(validation_errors, diffs, field_distance_breakdown):
    """Build a 'Mistakes to Avoid' section based on previous errors."""
    parts = []
    if validation_errors or diffs or field_distance_breakdown:
        lessons = generate_lessons_learned(field_distance_breakdown)
        if lessons:
            parts.append(f"\n\n{lessons}")
    if validation_errors:
        parts.append("\n\nValidation Errors to Fix:")
        for error in validation_errors:
            if isinstance(error, dict):
                parts.append(f"  - {json.dumps(error)}")
            else:
                parts.append(f"  - {error}")
    return "".join(parts) if parts else ""


def _is_array_field_in_schema(field_path, schema):
    """Check if the final field in a path is defined as array type in the schema."""
    if not schema or "properties" not in schema:
        return False
    clean_path = re.sub(r'\[\d+\]', '', field_path)
    path_parts = clean_path.split(".")
    current_schema = schema.get("properties", {})
    for part in path_parts[:-1]:
        if part not in current_schema:
            return False
        field_schema = current_schema[part]
        if field_schema.get("type") == "array" and "items" in field_schema:
            current_schema = field_schema["items"].get("properties", {})
        elif field_schema.get("type") == "object" and "properties" in field_schema:
            current_schema = field_schema["properties"]
        elif "$ref" in field_schema:
            return False
        else:
            return False
    final_field = path_parts[-1]
    if final_field not in current_schema:
        return False
    return current_schema[final_field].get("type") == "array"


def generate_lessons_learned(field_distance_breakdown, schema=None):
    """Convert field distance breakdown into generic lessons-learned bullet points."""
    if not field_distance_breakdown:
        return ""

    lessons = []
    missing_field_names = []
    empty_field_names = []
    value_mismatch_field_names = []
    actual_array_fields_with_issues = set()

    for field_path, breakdown in field_distance_breakdown.items():
        distance = breakdown.get("distance", 0)
        metric_type = breakdown.get("metric_type", "unknown")
        extracted = breakdown.get("extracted", "")

        if distance < 0.1:
            continue

        clean_path = re.sub(r'\[\d+\]', '', field_path)
        field_name = clean_path.split(".")[-1].strip("[]'\"")

        is_actually_array = schema and _is_array_field_in_schema(field_path, schema)
        has_array_indices = "[" in field_path
        is_true_array_field = is_actually_array and has_array_indices

        if metric_type == "missing_in_extracted":
            missing_field_names.append(field_name)
            if is_true_array_field:
                actual_array_fields_with_issues.add(field_name)
        elif extracted == "" or extracted is None:
            empty_field_names.append(field_name)
            if is_true_array_field:
                actual_array_fields_with_issues.add(field_name)
        else:
            value_mismatch_field_names.append(field_name)
            if is_true_array_field:
                actual_array_fields_with_issues.add(field_name)

    if missing_field_names or empty_field_names:
        problematic = set(missing_field_names + empty_field_names)
        if problematic:
            field_list = ", ".join(f'"{f}"' for f in sorted(problematic)[:3])
            if len(problematic) > 3:
                field_list += f", and {len(problematic) - 3} others"
            lessons.append(
                f"Ensure these fields are never left empty: {field_list}. "
                "Always extract or reasonably infer a value from the provided text."
            )

    if actual_array_fields_with_issues:
        array_list = ", ".join(f'"{f}"' for f in sorted(actual_array_fields_with_issues))
        lessons.append(
            f"Array fields like {array_list} must include ALL items found in the text. "
            "Do not skip or truncate array items."
        )

    if value_mismatch_field_names:
        lessons.append(
            "Ensure extracted values match the semantic meaning intended in the text. "
            "Pay attention to context and relationships between fields."
        )

    lessons.append(
        "Always validate that your extracted values conform to the schema definitions "
        "(field types, required fields, allowed values)."
    )
    lessons.append(
        "Extract information directly from the provided text. Do not invent or assume "
        "values not present in the source material. If information is unavailable, "
        "reason about the most appropriate extraction."
    )

    if not lessons:
        return ""
    lessons_text = "\n".join([f"* {lesson}" for lesson in lessons])
    return f"Lessons Learned from Previous Steps (for other text examples):\n{lessons_text}"
