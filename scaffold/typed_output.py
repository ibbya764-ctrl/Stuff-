"""
typed_output.py
===============

Defensive, retrying, schema-validated JSON output from any chat backend.

This is the load-bearing infrastructure for running the scaffold on
small local models. The single biggest difference between a 7B local
model and a frontier API is reliability of structured output: small
models hallucinate keys, drop closing braces, wrap output in extra
prose, and occasionally emit YAML instead of JSON. Without a layer
that handles all of that defensively, the rest of the pipeline breaks
constantly.

Three levels of robustness, used in increasing order of cost:

  1. Extract JSON from prose (strip fences, find balanced braces).
  2. Schema-check (required keys, types, length).
  3. Repair-by-retry: feed the bad output back with the schema and
     ask the model to fix it, up to N times.

Usage from the pipeline:

    from typed_output import typed_call, JsonSchema

    schema = JsonSchema(
        required={"branches": list},
        item_schema={"name": str, "method": str, "steps": list},
        min_items=2,
    )

    result = typed_call(
        llm_chat_fn,
        system=BRANCH_GEN_SYSTEM,
        user=question,
        schema=schema,
        tag="branch_gen",
        max_retries=2,
    )
    # result is either a parsed dict matching the schema, or None
    # after exhausting retries.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, Union


# ============================================================
# Schema definition
# ============================================================

JSON_TYPE = Union[Type, tuple]


@dataclass
class JsonSchema:
    """
    Lightweight schema for the kind of JSON the pipeline expects.

    `required`    : top-level keys → expected Python type
                    (e.g. {"branches": list, "summary": str})
    `optional`    : same shape, but absence is OK
    `item_schema` : if a required key is `list`, each item must match
                    this schema (recursively, one level deep)
    `min_items`   : if a key is list, minimum length
    `max_items`   : if a key is list, maximum length
    """

    required:    dict[str, JSON_TYPE]              = field(default_factory=dict)
    optional:    dict[str, JSON_TYPE]              = field(default_factory=dict)
    item_schema: Optional[dict[str, JSON_TYPE]]    = None
    min_items:   Optional[int]                     = None
    max_items:   Optional[int]                     = None

    def describe(self) -> str:
        """Human-readable summary used in repair prompts."""
        lines = ["Required keys:"]
        for k, t in self.required.items():
            type_name = (t.__name__ if hasattr(t, "__name__") else str(t))
            lines.append(f"  - {k!r} : {type_name}")
        if self.optional:
            lines.append("Optional keys:")
            for k, t in self.optional.items():
                type_name = (t.__name__ if hasattr(t, "__name__") else str(t))
                lines.append(f"  - {k!r} : {type_name}")
        if self.item_schema:
            lines.append("Each list item must have:")
            for k, t in self.item_schema.items():
                type_name = (t.__name__ if hasattr(t, "__name__") else str(t))
                lines.append(f"    - {k!r} : {type_name}")
        if self.min_items is not None:
            lines.append(f"Minimum items in list: {self.min_items}")
        if self.max_items is not None:
            lines.append(f"Maximum items in list: {self.max_items}")
        return "\n".join(lines)


# ============================================================
# Layer 1 — extract JSON from arbitrary prose
# ============================================================

_JSON_FENCE_RE   = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)```")
_OBJECT_FIND_RE  = re.compile(r"\{[\s\S]*\}")
_TRAILING_COMMA  = re.compile(r",(\s*[}\]])")
_PYTHON_NONE     = re.compile(r"\bNone\b")
_PYTHON_TRUE     = re.compile(r"\bTrue\b")
_PYTHON_FALSE    = re.compile(r"\bFalse\b")


def extract_json(text: str) -> Optional[Any]:
    """
    Pull a JSON object/array out of arbitrary prose.
    Returns the parsed Python object, or None if extraction failed.

    Robust to:
      - markdown fences ```json ... ```
      - leading/trailing prose
      - trailing commas
      - python-style None/True/False
      - single quotes (heuristic only — last resort)
    """
    if not isinstance(text, str) or not text.strip():
        return None

    candidates: list[str] = []

    # 1. Try the body of a fenced block first.
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1))

    # 2. Try the largest balanced { ... } block.
    obj_match = _OBJECT_FIND_RE.search(text)
    if obj_match:
        candidates.append(obj_match.group(0))

    # 3. Try the raw text last.
    candidates.append(text.strip())

    for cand in candidates:
        # Direct parse
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass

        # Cleanup pass: kill trailing commas + python-isms
        cleaned = cand
        cleaned = _TRAILING_COMMA.sub(r"\1", cleaned)
        cleaned = _PYTHON_NONE.sub("null", cleaned)
        cleaned = _PYTHON_TRUE.sub("true", cleaned)
        cleaned = _PYTHON_FALSE.sub("false", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Last-resort single-quote replacement (heuristic, only if no
        # apostrophes inside string-like spans).
        if "'" in cleaned and '"' not in cleaned:
            try:
                return json.loads(cleaned.replace("'", '"'))
            except json.JSONDecodeError:
                pass

    return None


# ============================================================
# Layer 2 — validate against schema
# ============================================================

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate(obj: Any, schema: JsonSchema) -> ValidationResult:
    """Check `obj` against `schema`. Returns errors that can be fed
    back to the model in a repair prompt."""
    errors: list[str] = []

    if not isinstance(obj, dict):
        return ValidationResult(False,
                                [f"Top-level must be a JSON object, got {type(obj).__name__}"])

    for key, expected_type in schema.required.items():
        if key not in obj:
            errors.append(f"Missing required key {key!r}")
            continue
        if not _isinstance_lenient(obj[key], expected_type):
            errors.append(
                f"Key {key!r} should be {_typename(expected_type)}, "
                f"got {type(obj[key]).__name__}"
            )

    for key, expected_type in schema.optional.items():
        if key in obj and not _isinstance_lenient(obj[key], expected_type):
            errors.append(
                f"Optional key {key!r} should be {_typename(expected_type)} "
                f"if present, got {type(obj[key]).__name__}"
            )

    # Item-level checks on the first list-typed required key
    if schema.item_schema is not None:
        list_keys = [k for k, t in schema.required.items() if t is list]
        for key in list_keys:
            if key not in obj or not isinstance(obj[key], list):
                continue
            items = obj[key]
            if schema.min_items is not None and len(items) < schema.min_items:
                errors.append(f"Key {key!r} needs at least {schema.min_items} "
                              f"items (got {len(items)})")
            if schema.max_items is not None and len(items) > schema.max_items:
                errors.append(f"Key {key!r} has too many items "
                              f"(max {schema.max_items}, got {len(items)})")
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(f"{key}[{i}] must be an object, "
                                  f"got {type(item).__name__}")
                    continue
                for sub_key, sub_type in schema.item_schema.items():
                    if sub_key not in item:
                        errors.append(f"{key}[{i}] missing key {sub_key!r}")
                    elif not _isinstance_lenient(item[sub_key], sub_type):
                        errors.append(
                            f"{key}[{i}][{sub_key!r}] should be "
                            f"{_typename(sub_type)}, "
                            f"got {type(item[sub_key]).__name__}"
                        )

    return ValidationResult(valid=(not errors), errors=errors)


def _isinstance_lenient(value, expected: JSON_TYPE) -> bool:
    """Accept None for any type if the field is plausibly optional."""
    if value is None:
        return False
    if isinstance(expected, tuple):
        return isinstance(value, expected)
    # str/int distinction: accept "5" for int? No — be strict here so
    # the model is told the right type.
    return isinstance(value, expected)


def _typename(t: JSON_TYPE) -> str:
    if isinstance(t, tuple):
        return " or ".join(x.__name__ for x in t)
    return t.__name__ if hasattr(t, "__name__") else str(t)


# ============================================================
# Layer 3 — repair-by-retry
# ============================================================

REPAIR_SYSTEM = """You are repairing malformed JSON output.

The previous output had errors. Re-emit ONLY a corrected JSON object
matching the required schema. No prose, no markdown fences, no comments.
"""


def _build_repair_user(
    schema: JsonSchema,
    bad_output: str,
    errors: list[str],
) -> str:
    return (
        "The output you produced did not match the required schema.\n\n"
        f"Schema:\n{schema.describe()}\n\n"
        f"Errors found:\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\n\nYour previous (invalid) output was:\n"
        + bad_output[:4000]
        + "\n\nReturn only the corrected JSON object. No prose."
    )


# ============================================================
# Public API
# ============================================================

@dataclass
class TypedCallResult:
    """Result of a typed call. Always populated even on failure."""
    value:          Optional[dict]
    valid:          bool
    attempts:       int
    errors:         list[str] = field(default_factory=list)
    raw_responses:  list[str] = field(default_factory=list)
    total_time_s:   float = 0.0

    def __bool__(self) -> bool:
        return self.valid


def typed_call(
    llm_chat_fn: Callable[[str, str], str],
    system: str,
    user: str,
    schema: JsonSchema,
    *,
    max_retries: int = 2,
    tag: str = "",
    on_retry: Optional[Callable[[int, list[str]], None]] = None,
) -> TypedCallResult:
    """
    Call the LLM and return a parsed, schema-validated JSON object.

    Behaviour:
      1. Call the model.
      2. Extract JSON from the response.
      3. Validate against schema.
      4. If invalid: feed the bad output + schema + errors back in a
         repair prompt. Retry up to `max_retries` times.
      5. Return TypedCallResult with the parsed object (or None) plus
         provenance (attempts, errors, raw responses).

    Pass `tag` through so telemetry can categorise calls.
    """
    t0 = time.time()
    raw_responses: list[str] = []
    last_errors: list[str] = []

    # ---- Attempt 1: original call ----
    response = _safe_call(llm_chat_fn, system, user)
    raw_responses.append(response)
    obj = extract_json(response)
    if obj is None:
        last_errors = ["Could not extract any JSON object from response"]
    else:
        result = validate(obj, schema)
        if result.valid:
            return TypedCallResult(
                value=obj, valid=True, attempts=1,
                raw_responses=raw_responses,
                total_time_s=time.time() - t0,
            )
        last_errors = result.errors

    # ---- Attempts 2..N: repair ----
    for attempt in range(2, max_retries + 2):
        if on_retry is not None:
            try:
                on_retry(attempt - 1, last_errors)
            except Exception:
                pass

        repair_user = _build_repair_user(schema, response, last_errors)
        response = _safe_call(llm_chat_fn, REPAIR_SYSTEM, repair_user)
        raw_responses.append(response)
        obj = extract_json(response)
        if obj is None:
            last_errors = ["Repair attempt could not produce parseable JSON"]
            continue
        result = validate(obj, schema)
        if result.valid:
            return TypedCallResult(
                value=obj, valid=True, attempts=attempt,
                errors=last_errors, raw_responses=raw_responses,
                total_time_s=time.time() - t0,
            )
        last_errors = result.errors

    return TypedCallResult(
        value=None, valid=False, attempts=max_retries + 1,
        errors=last_errors, raw_responses=raw_responses,
        total_time_s=time.time() - t0,
    )


def _safe_call(llm_chat_fn, system, user) -> str:
    try:
        return llm_chat_fn(system, user) or ""
    except Exception as e:
        return f"[call failed: {e}]"


# ============================================================
# Pre-built schemas for the pipeline's main call types
# ============================================================

BRANCHES_SCHEMA = JsonSchema(
    required={"branches": list},
    item_schema={
        "name":             str,
        "method":           str,
        "steps":            list,
        "assumptions":      list,
        "candidate_result": str,
    },
    min_items=1,
    max_items=10,
)


AUDIT_QUESTIONS_SCHEMA = JsonSchema(
    required={"questions": list},
    item_schema={
        "question": str,
        "targets":  str,
    },
    min_items=1,
    max_items=10,
)


CRITIQUE_SCHEMA = JsonSchema(
    required={"evaluations": list},
    item_schema={
        "index":   int,
        "verdict": str,
        "reason":  str,
    },
    min_items=1,
)


GRAPH_EXTRACTION_SCHEMA = JsonSchema(
    required={"entities": list, "relations": list},
    item_schema={"id": str, "type": str},   # entities
    min_items=1,
)


CURIOSITY_QUESTION_SCHEMA = JsonSchema(
    required={"question": str},
    optional={"rationale": str},
)
