"""JSON parsing, flattening, and blacklist utilities."""

import json
import logging
import re

from optimize_prompt_json.config import BLACKLIST_FIELDS

logger = logging.getLogger(__name__)


def is_field_blacklisted(field_name):
    """Check if a field name is in the blacklist (case-insensitive)."""
    return field_name.lower() in {f.lower() for f in BLACKLIST_FIELDS}


def remove_blacklisted_fields(obj):
    """Recursively remove blacklisted fields from a JSON object."""
    if isinstance(obj, dict):
        return {
            k: remove_blacklisted_fields(v)
            for k, v in obj.items()
            if not is_field_blacklisted(k)
        }
    elif isinstance(obj, list):
        return [remove_blacklisted_fields(item) for item in obj]
    return obj


def build_exclude_paths_from_blacklist(obj, field_blacklist=None, prefix=""):
    """Build DeepDiff-compatible exclude paths from blacklisted field names."""
    if field_blacklist is None:
        field_blacklist = {f.lower() for f in BLACKLIST_FIELDS}
    if not field_blacklist:
        return []
    exclude_paths = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{prefix}['{key}']" if prefix else f"root['{key}']"
            if key.lower() in field_blacklist:
                exclude_paths.append(current_path)
            else:
                exclude_paths.extend(
                    build_exclude_paths_from_blacklist(value, field_blacklist, current_path)
                )
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            current_path = f"{prefix}[{i}]"
            exclude_paths.extend(
                build_exclude_paths_from_blacklist(item, field_blacklist, current_path)
            )
    return exclude_paths


def flatten_json(obj, path=""):
    """Flatten a JSON object into a set of (path, value) tuples."""
    items = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            items |= flatten_json(v, new_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items |= flatten_json(v, f"{path}[{i}]")
    else:
        items.add((path, str(obj)))
    return items


def extract_arrays_from_json(obj, path=""):
    """Extract all scalar array fields from JSON with their element lists."""
    arrays = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            arrays.update(extract_arrays_from_json(v, new_path))
    elif isinstance(obj, list):
        if obj and all(not isinstance(item, (dict, list)) for item in obj):
            arrays[path] = [str(item) for item in obj]
        for i, v in enumerate(obj):
            arrays.update(extract_arrays_from_json(v, f"{path}[{i}]"))
    return arrays


def _escape_control_chars_in_json_strings(s):
    """Escape literal control characters inside JSON string values."""
    result = []
    in_string = False
    escape_next = False
    for char in s:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        if in_string:
            if char == '\n':
                result.append('\\n')
                continue
            if char == '\r':
                result.append('\\r')
                continue
            if char == '\t':
                result.append('\\t')
                continue
        result.append(char)
    return ''.join(result)


def extract_json_from_response(input_string):
    """Extract and parse the first valid JSON object or array from a string."""
    input_string = re.sub(r'```(?:json|python|text)?\s*', '', input_string, flags=re.IGNORECASE)
    input_string = re.sub(r'\s*```', '', input_string)
    input_string = input_string.strip()

    start_match = re.search(r'[\[\{]', input_string)
    end_match = re.search(r'[\]\}]', input_string[::-1])
    if not start_match or not end_match:
        logger.warning("No JSON boundaries found in input string.")
        return None

    start_index = start_match.start()
    end_index = len(input_string) - end_match.start()
    json_candidate = input_string[start_index:end_index]

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_escape_control_chars_in_json_strings(json_candidate))
    except json.JSONDecodeError:
        pass

    for m in re.findall(r'([\{\[].*[\}\]])', input_string, flags=re.DOTALL):
        try:
            return json.loads(m)
        except Exception:
            pass
        try:
            return json.loads(_escape_control_chars_in_json_strings(m))
        except Exception:
            continue

    logger.error("Failed to decode JSON after regex fallback.")
    return None
