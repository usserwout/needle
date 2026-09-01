"""Dataset, validation, and evaluation helpers for Dutch Needle fine-tuning.

The module deliberately has no JAX dependency.  It can therefore be used to
prepare and audit data on a regular Python installation before a training
environment is created.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


_ANNOTATION = re.compile(r"\[\s*([^\]:]+?)\s*:\s*([^\]]+?)\s*\]")
_NORMALIZE_QUERY = re.compile(r"[^\w]+", re.UNICODE)
_VERBS = {
    "add", "cancel", "change", "check", "create", "delete", "find",
    "get", "increase", "lower", "pause", "play", "remove", "search",
    "send", "set", "show", "start", "stop", "turn", "update",
}


def _as_calls(value: Any) -> list[dict[str, Any]]:
    """Convert an answer/call envelope into a consistently ordered call list."""
    if isinstance(value, dict):
        value = value.get("function_calls", value.get("answers", []))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    calls = []
    for call in value:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            continue
        arguments = call.get("arguments") or {}
        calls.append({"name": call["name"], "arguments": arguments})
    return calls


def canonical_calls(value: Any) -> str:
    """Canonical JSON representation used for semantic exact-call matching."""
    return json.dumps(_as_calls(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _normal_query(query: str) -> str:
    return _NORMALIZE_QUERY.sub(" ", query.casefold()).strip()


def _json_type_ok(value: Any, schema: dict[str, Any]) -> bool:
    typ = schema.get("type")
    if typ == "string":
        return isinstance(value, str)
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ == "array":
        return isinstance(value, list)
    if typ == "object":
        return isinstance(value, dict)
    return True


def _validate_value(name: str, value: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _json_type_ok(value, schema):
        return [f"argument {name!r} has wrong type"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"argument {name!r} is outside enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"argument {name!r} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"argument {name!r} is above maximum")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"argument {name!r} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"argument {name!r} is too long")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"argument {name!r} does not match pattern")
    return errors


def validate_example(example: dict[str, Any], *, require_grounding: bool = False) -> list[str]:
    """Return human-readable contract violations for one Needle JSONL record."""
    errors: list[str] = []
    query = example.get("query")
    if not isinstance(query, str) or not query.strip():
        return ["query must be a non-empty string"]
    tools = example.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return ["tools is not valid JSON"]
    if not isinstance(tools, list):
        return ["tools must be a list"]
    by_name = {tool.get("name"): tool for tool in tools if isinstance(tool, dict)}
    if len(by_name) != len(tools) or any(not isinstance(name, str) for name in by_name):
        errors.append("tools must have unique string names")
    answers = example.get("answers", example.get("function_calls", []))
    if not isinstance(answers, list):
        return errors + ["answers must be a list"]
    query_folded = query.casefold()
    for position, call in enumerate(answers):
        if not isinstance(call, dict):
            errors.append(f"answer {position} must be an object")
            continue
        name = call.get("name")
        tool = by_name.get(name)
        if tool is None:
            errors.append(f"answer {position} references unknown tool {name!r}")
            continue
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            errors.append(f"answer {position} arguments must be an object")
            continue
        parameters = tool.get("parameters", {})
        if not isinstance(parameters, dict):
            errors.append(f"tool {name!r} parameters must be an object")
            continue
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"tool {name!r} properties must be an object")
            continue
        required = parameters.get("required", [])
        if not isinstance(required, list):
            errors.append(f"tool {name!r} required must be a list")
            continue
        for field in required:
            if field not in arguments:
                errors.append(f"answer {position} misses required argument {field!r}")
        for field, value in arguments.items():
            if field not in properties:
                errors.append(f"answer {position} has unknown argument {field!r}")
                continue
            if not isinstance(properties[field], dict):
                errors.append(f"tool {name!r} property {field!r} must be an object")
                continue
            errors.extend(_validate_value(field, value, properties[field]))
            if (require_grounding and isinstance(value, str) and len(value.strip()) > 1
                    and value.casefold() not in query_folded):
                errors.append(f"argument {field!r} is not grounded in query")
    return errors


def validate_jsonl(path: str | Path, *, require_grounding: bool = False) -> dict[str, Any]:
    """Validate JSONL examples and flag exact normalized queries across splits."""
    path = Path(path)
    errors: list[dict[str, Any]] = []
    by_query: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    by_group: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    by_token_set: dict[tuple[str, ...], list[tuple[int, str, str]]] = collections.defaultdict(list)
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_no, "errors": [f"invalid JSON: {exc.msg}"]})
                continue
            row_errors = validate_example(example, require_grounding=require_grounding)
            if row_errors:
                errors.append({"line": line_no, "errors": row_errors})
            if isinstance(example.get("query"), str):
                normalized = _normal_query(example["query"])
                split = str(example.get("split", "unspecified"))
                by_query[normalized].append(
                    (line_no, split))
                tokens = tuple(sorted(set(normalized.split())))
                if len(tokens) >= 3:
                    by_token_set[tokens].append((line_no, split, normalized))
            if example.get("group_id"):
                by_group[str(example["group_id"])].append(
                    (line_no, str(example.get("split", "unspecified"))))
    duplicates = []
    for query, locations in by_query.items():
        splits = sorted({split for _, split in locations})
        if query and len(splits) > 1:
            duplicates.append({"query": query, "splits": splits,
                               "lines": [line for line, _ in locations]})
    group_leaks = []
    for group_id, locations in by_group.items():
        splits = sorted({split for _, split in locations})
        if len(splits) > 1:
            group_leaks.append({"group_id": group_id, "splits": splits,
                                "lines": [line for line, _ in locations]})
    near_duplicates = []
    for tokens, locations in by_token_set.items():
        splits = sorted({split for _, split, _ in locations})
        queries = {query for _, _, query in locations}
        if len(splits) > 1 and len(queries) > 1:
            near_duplicates.append({"tokens": list(tokens), "splits": splits,
                                    "lines": [line for line, _, _ in locations]})
    return {
        "path": str(path), "examples": total, "invalid_examples": len(errors),
        "errors": errors, "cross_split_duplicates": duplicates,
        "cross_split_group_leaks": group_leaks,
        "near_cross_split_duplicates": near_duplicates,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def validate_many_jsonl(paths: Sequence[str | Path], *, require_grounding: bool = False) -> dict[str, Any]:
    """Validate several split files and make cross-file leakage a hard error."""
    reports = [validate_jsonl(path, require_grounding=require_grounding) for path in paths]
    locations: dict[str, list[tuple[str, str, int, str]]] = collections.defaultdict(list)
    groups: dict[str, list[tuple[str, str, int]]] = collections.defaultdict(list)
    token_sets: dict[tuple[str, ...], list[tuple[str, str, int, str]]] = collections.defaultdict(list)
    for path in paths:
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                split = str(row.get("split", path.stem))
                if isinstance(row.get("query"), str):
                    normalized = _normal_query(row["query"])
                    locations[normalized].append(
                        (str(path), split, line_no, str(row.get("source", ""))))
                    tokens = tuple(sorted(set(normalized.split())))
                    if len(tokens) >= 3:
                        token_sets[tokens].append(
                            (str(path), split, line_no, normalized, str(row.get("source", ""))))
                if row.get("group_id"):
                    groups[str(row["group_id"])].append((str(path), split, line_no))

    def cross_split(values):
        return len({split for _, split, *_ in values}) > 1

    exact = [{"query": query, "locations": values} for query, values in locations.items()
             if query and cross_split(values)]
    group_leaks = [{"group_id": group_id, "locations": values} for group_id, values in groups.items()
                   if cross_split(values)]
    near = [{"tokens": list(tokens), "locations": values} for tokens, values in token_sets.items()
            if len({value[3] for value in values}) > 1 and cross_split(values)]
    # MASSIVE itself contains a small number of repeated utterances between its
    # immutable dev/test partitions.  Report those, but only fail the gate for
    # leakage involving training data; the frozen evaluation partitions cannot
    # be edited without changing the benchmark.
    def touches_training(item: dict[str, Any]) -> bool:
        return any(str(location[1]) == "train" for location in item["locations"])

    def official_massive_duplicate(item: dict[str, Any]) -> bool:
        locations = item["locations"]
        return bool(locations) and all(
            (location[4] if len(location) >= 5 else location[3])
            == "MASSIVE-1.0-CC-BY-4.0"
            for location in locations
        )

    train_exact = [
        item for item in exact
        if touches_training(item) and not official_massive_duplicate(item)
    ]
    train_groups = [item for item in group_leaks if touches_training(item)]
    train_near = [
        item for item in near
        if touches_training(item) and not official_massive_duplicate(item)
    ]
    invalid = sum(report["invalid_examples"] for report in reports)
    return {
        "files": [report["path"] for report in reports], "examples": sum(report["examples"] for report in reports),
        "invalid_examples": invalid, "reports": reports,
        "cross_split_duplicates": exact, "cross_split_group_leaks": group_leaks,
        "near_cross_split_duplicates": near,
        "training_leakage": {
            "duplicates": train_exact, "group_leaks": train_groups, "near_duplicates": train_near,
        },
        "evaluation_source_duplicates": {
            "duplicates": [
                item for item in exact
                if not touches_training(item) or official_massive_duplicate(item)
            ],
            "group_leaks": [item for item in group_leaks if not touches_training(item)],
            "near_duplicates": [item for item in near if not touches_training(item)],
        },
        "leakage_examples": len(train_exact) + len(train_groups) + len(train_near),
    }


def _safe_tool_name(intent: str) -> str:
    parts = [part for part in intent.lower().split("_") if part]
    if len(parts) > 1 and parts[-1] in _VERBS:
        return "_".join([parts[-1], *parts[:-1]])
    return "_".join(parts)


def _dutch_variant(query: str, position: int) -> tuple[str, str]:
    """Apply small, label-preserving Dutch register variants to a query."""
    variant = position % 10
    if variant < 7:
        return query, "nl-NL"
    if variant < 9:
        return "Kunt ge dit doen? " + query, "nl-BE"
    noisy = query.casefold().replace("kun je", "kun je ff").replace("een ", "n ")
    return "pls " + noisy, "nl-noisy"


def _slots(annotated: str | None) -> dict[str, str]:
    return {name.strip(): value.strip() for name, value in _ANNOTATION.findall(annotated or "")}


def _balanced_rows(rows: Sequence[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[str(row["intent"])].append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < limit:
        progressed = False
        for intent in sorted(groups):
            if groups[intent] and len(selected) < limit:
                selected.append(groups[intent].pop())
                progressed = True
        if not progressed:
            break
    return selected


def build_massive_examples(rows: Iterable[dict[str, Any]], *, partition: str,
                           limit: int = 0, seed: int = 0,
                           split: str | None = None) -> list[dict[str, Any]]:
    """Convert labelled MASSIVE rows to schema-conditioned Needle examples.

    The input may be the official ``nl-NL.jsonl`` file or already decoded rows.
    Tool schemas are derived from annotations, retaining English identifiers while
    preserving the Dutch utterance and slot values.
    """
    all_rows = [row for row in rows if row.get("intent") and row.get("utt")]
    chosen = [row for row in all_rows if row.get("partition") == partition]
    chosen = _balanced_rows(chosen, limit, seed)
    slots_by_intent: dict[str, set[str]] = collections.defaultdict(set)
    scenario_intents: dict[str, list[str]] = collections.defaultdict(list)
    for row in all_rows:
        intent = str(row["intent"])
        slots_by_intent[intent].update(_slots(row.get("annot_utt")).keys())
        scenario = str(row.get("scenario") or "general")
        if intent not in scenario_intents[scenario]:
            scenario_intents[scenario].append(intent)

    def schema(intent: str, variant: int) -> dict[str, Any]:
        fields = sorted(slots_by_intent[intent])
        readable_intent = intent.replace("_", " ")
        descriptions = [
            "Handle a user's " + readable_intent + " request.",
            "Use this tool when a user wants to " + readable_intent + ".",
            "Perform the requested " + readable_intent + " action.",
        ]
        return {
            "name": _safe_tool_name(intent),
            "description": descriptions[variant],
            "parameters": {
                "type": "object",
                "properties": {field: {"type": "string", "description": field.replace("_", " ")}
                               for field in fields},
                # MASSIVE labels only the slots spoken in each utterance.  Keeping
                # the union optional prevents invented values for absent slots.
                "required": [],
            },
        }

    rng = random.Random(seed)
    examples = []
    for position, row in enumerate(chosen):
        intent = str(row["intent"])
        scenario = str(row.get("scenario") or "general")
        candidates = list(scenario_intents[scenario])
        rng.shuffle(candidates)
        candidates = [intent] + [other for other in candidates if other != intent]
        if len(candidates) < 3:
            filler = [other for other in sorted(slots_by_intent) if other not in candidates]
            rng.shuffle(filler)
            candidates.extend(filler[:3 - len(candidates)])
        # Four candidates retain semantically similar distractors while
        # keeping the largest MASSIVE schemas within the 1,024-token budget.
        # Never rely on truncation to fit a catalogue into the prompt.
        candidates = candidates[:4]
        rng.shuffle(candidates)
        arguments = _slots(row.get("annot_utt"))
        if row.get("locale") == "nl-NL":
            query, locale = _dutch_variant(str(row["utt"]), position)
        else:
            query, locale = row["utt"], row.get("locale", "en-US")
        # Some localized MASSIVE annotations contain a slot surface form that
        # differs from the raw utterance.  Do not teach an ungrounded value:
        # omitted fields remain optional in the generated schema.
        query_folded = query.casefold()
        arguments = {
            field: value for field, value in arguments.items()
            if value.casefold() in query_folded
        }
        schema_variant = 2 if partition == "test" else position % 2
        reasoning = "; ".join(
            f"{field}='{value}' uit de vraag" for field, value in sorted(arguments.items())
        )
        examples.append({
            "id": "massive:" + str(row.get("id", len(examples))),
            "group_id": "massive:" + str(row.get("id", len(examples))),
            "source": "MASSIVE-1.0-CC-BY-4.0",
            "locale": locale, "domain": scenario, "schema_variant": schema_variant,
            "tool_similarity": "same_scenario", "task_family": "action",
            "split": split or partition,
            "query": query,
            "tools": [schema(candidate, schema_variant) for candidate in candidates],
            "answers": [{"name": _safe_tool_name(intent), "arguments": arguments}],
            "reasoning": reasoning,
        })
    return examples


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _slice_report(rows: Sequence[dict[str, Any]], exact: Sequence[bool], field: str) -> dict[str, Any]:
    buckets: dict[str, list[bool]] = collections.defaultdict(list)
    for row, value in zip(rows, exact):
        buckets[str(row.get(field, "unspecified"))].append(value)
    return {name: {"examples": len(values), "exact_call_accuracy": sum(values) / len(values)}
            for name, values in sorted(buckets.items())}


def _query_length_bucket(row: dict[str, Any]) -> str:
    words = len(str(row.get("query", "")).split())
    if words < 8:
        return "short"
    if words < 20:
        return "medium"
    return "long"


def _noise_bucket(row: dict[str, Any]) -> str:
    return "noisy" if row.get("locale") == "nl-noisy" else "clean"


def score_predictions(rows: Sequence[dict[str, Any]], predictions: Sequence[Any], *,
                      bootstrap_samples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Score decoded calls against JSONL examples without requiring model libraries."""
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length")
    expected = [_as_calls(row.get("answers", row.get("function_calls", []))) for row in rows]
    actual = [_as_calls(prediction) for prediction in predictions]
    exact = [canonical_calls(want) == canonical_calls(got) for want, got in zip(expected, actual)]
    positives = [bool(want) for want in expected]
    predicted_positive = [bool(got) for got in actual]
    tp = sum(want and got for want, got in zip(positives, predicted_positive))
    fp = sum(not want and got for want, got in zip(positives, predicted_positive))
    fn = sum(want and not got for want, got in zip(positives, predicted_positive))
    no_tp = sum(not want and not got for want, got in zip(positives, predicted_positive))
    no_fp = fn
    no_fn = fp
    tool_correct = 0
    argument_correct = 0
    argument_cases = 0
    hallucinated = 0
    generated_arguments = 0
    multi_indices = []
    for index, (row, want, got) in enumerate(zip(rows, expected, actual)):
        if len(want) > 1:
            multi_indices.append(index)
        if want and got and [call["name"] for call in want] == [call["name"] for call in got]:
            tool_correct += 1
            argument_cases += 1
            if all(left["arguments"] == right["arguments"] for left, right in zip(want, got)):
                argument_correct += 1
        query = str(row.get("query", "")).casefold()
        for call in got:
            for value in call["arguments"].values():
                if isinstance(value, str) and value.strip():
                    generated_arguments += 1
                    if value.casefold() not in query:
                        hallucinated += 1
    rng = random.Random(seed)
    estimates = []
    if exact and bootstrap_samples:
        for _ in range(bootstrap_samples):
            estimates.append(sum(exact[rng.randrange(len(exact))] for _ in exact) / len(exact))
        estimates.sort()
        low = estimates[max(0, math.floor(0.025 * len(estimates)))]
        high = estimates[min(len(estimates) - 1, math.ceil(0.975 * len(estimates)) - 1)]
    else:
        low = high = 0.0
    return {
        "examples": len(rows),
        "exact_call_accuracy": _ratio(sum(exact), len(exact)),
        "exact_call_ci95": [low, high],
        "tool_selection_accuracy": _ratio(tool_correct, sum(positives)),
        "argument_exact_match": _ratio(argument_correct, argument_cases),
        "hallucinated_argument_rate": _ratio(hallucinated, generated_arguments),
        "no_call": {
            "precision": _ratio(no_tp, no_tp + no_fp),
            "recall": _ratio(no_tp, no_tp + no_fn),
            "f1": _f1(_ratio(no_tp, no_tp + no_fp), _ratio(no_tp, no_tp + no_fn)),
        },
        "multi_call_exact_match": _ratio(sum(exact[i] for i in multi_indices), len(multi_indices)),
        "slices": {
            "task_family": _slice_report(rows, exact, "task_family"),
            "locale": _slice_report(rows, exact, "locale"),
            "source": _slice_report(rows, exact, "source"),
            "domain": _slice_report(rows, exact, "domain"),
            "extraction_type": _slice_report(rows, exact, "extraction_type"),
            "tool_similarity": _slice_report(rows, exact, "tool_similarity"),
            "schema_variant": _slice_report(rows, exact, "schema_variant"),
            "query_length": _slice_report(
                [{**row, "query_length": _query_length_bucket(row)} for row in rows],
                exact, "query_length"),
            "noise": _slice_report(
                [{**row, "noise": _noise_bucket(row)} for row in rows], exact, "noise"),
        },
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records with line-aware errors."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return str(path)


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_EXTRACTION_SCHEMAS = [
    ("extract_contact", ("name", "email", "phone")),
    ("extract_appointment", ("date", "time", "location")),
    ("extract_invoice", ("vendor", "invoice_number", "total")),
    ("extract_order", ("order_number", "product", "quantity")),
    ("extract_shipment", ("tracking_number", "carrier", "status")),
    ("extract_product", ("product", "price", "availability")),
    ("extract_citation", ("author", "title", "year")),
    ("extract_form", ("name", "address", "postal_code")),
    ("extract_message", ("sender", "subject", "message")),
    ("extract_event", ("event", "date", "venue")),
    ("extract_receipt", ("merchant", "date", "total", "currency")),
    ("extract_expense", ("merchant", "date", "total", "category")),
    ("extract_shipping_label", ("recipient", "address", "tracking_number", "carrier")),
    ("extract_line_item", ("item", "quantity", "unit_price", "total")),
    ("extract_doctor_visit", ("doctor", "specialty", "date", "time")),
    ("extract_bank_transfer", ("recipient", "iban", "amount", "description")),
    ("extract_hotel_booking", ("hotel", "city", "checkin", "nights")),
    ("extract_flight", ("airline", "flight_number", "origin", "destination")),
    ("extract_customer_ticket", ("customer", "issue", "priority", "order_id")),
    ("extract_return_request", ("customer", "order_number", "reason", "item")),
]


def _extraction_tool(name: str, fields: Sequence[str], variant: int = 0) -> dict[str, Any]:
    readable = name.removeprefix("extract_").replace("_", " ")
    return {
        "name": name,
        "description": [
            "Extract " + readable + " fields from text.",
            "Read the text and return the " + readable + " values.",
            "Find the requested " + readable + " details in this passage.",
        ][variant % 3],
        "parameters": {
            "type": "object",
            "properties": {field: {"type": "string", "description": field.replace("_", " ")}
                           for field in fields},
            "required": list(fields),
        },
    }


def _candidate_extraction_tools(index: int, count: int = 3, variant: int = 0) -> list[dict[str, Any]]:
    choices = []
    for offset in range(count):
        name, fields = _EXTRACTION_SCHEMAS[(index + offset) % len(_EXTRACTION_SCHEMAS)]
        choices.append(_extraction_tool(name, fields, variant))
    return choices


def _dialect(index: int) -> str:
    remainder = index % 10
    if remainder < 7:
        return "nl-NL"
    if remainder < 9:
        return "nl-BE"
    return "nl-noisy"


def _extraction_values(name: str, fields: Sequence[str], index: int) -> dict[str, str]:
    first_names = [
        "Noor", "Bram", "Charlotte", "Daan", "Elise", "Fien", "Gijs", "Hanne",
        "Ilias", "Joris", "Kim", "Lotte", "Milan", "Nora", "Olivier", "Puck",
        "Ruben", "Sanne", "Thijs", "Yara", "Willem", "Emma", "Lars", "Anouk",
    ]
    last_names = [
        "de Vries", "Jansen", "de Wit", "Peeters", "Smit", "Vermeulen", "van Dijk",
        "Jacobs", "Bakker", "Willems", "Maes", "Visser", "Hendriks", "Claes",
        "de Boer", "Van den Bossche", "Wouters", "Hermans", "Vos", "Aarts",
    ]
    cities = [
        "Utrecht", "Gent", "Amsterdam", "Antwerpen", "Rotterdam", "Leuven",
        "Eindhoven", "Brugge", "Groningen", "Brussel", "Arnhem", "Mechelen",
    ]
    vendors = [
        "Voorbeeld BV", "Techniek Direct", "Kantoorshop Nederland", "BouwCenter Noord",
        "Groothandel De Valk", "MediaSolutions", "Logistiek Partners", "Bakkerij De Zwaan",
    ]
    products = [
        "draadloze koptelefoon", "koffiemolen", "e-reader", "usb-c-kabel",
        "regenjas", "wandelschoenen", "staande lamp", "steelpan",
        "notitieboek", "fietsslot", "webcam", "bluetoothspeaker",
    ]
    doctors = ["Dr. Peeters", "Dr. Van Dijk", "Dr. De Smet", "Dr. Bakker", "Dr. Visser", "Dr. Willems"]
    specialties = ["huisarts", "tandarts", "oogarts", "dermatoloog", "fysiotherapeut", "cardioloog"]
    hotels = ["Hotel De Korenbeurs", "Grand Hotel Karel V", "The Dominican", "Hotel Gravensteen"]
    airlines = ["KLM", "Brussels Airlines", "Transavia", "Lufthansa"]
    reasons = ["verkeerde maat", "beschadigd artikel", "defect bij levering", "niet naar verwachting"]
    issues = ["inloggen lukt niet", "betaling mislukt", "levering vertraagd", "retourzending niet verwerkt"]

    first = first_names[index % len(first_names)]
    last = last_names[(index // 3) % len(last_names)]
    full_name = f"{first} {last} {index}"
    city = cities[index % len(cities)]
    dest_city = cities[(index + 4) % len(cities)]

    values = {
        "name": full_name,
        "email": f"{first.lower()}{index}@voorbeeld.nl",
        "phone": f"+31 6 1234 {index % 9000 + 1000:04d}",
        "date": f"2026-10-{index % 28 + 1:02d}",
        "time": f"{8 + index % 10:02d}:30",
        "location": city,
        "vendor": f"{vendors[index % len(vendors)]} {index}",
        "invoice_number": f"INV-2026-{index % 90000 + 10000:05d}",
        "total": f"€ {index % 900 + 100},00",
        "order_number": f"ORD-{index % 900000 + 100000:06d}",
        "product": products[index % len(products)],
        "quantity": str(index % 5 + 1),
        "tracking_number": f"3SNL{index % 90000000000 + 10000000000:011d}",
        "carrier": ("PostNL" if index % 2 else "bpost"),
        "status": ("onderweg" if index % 2 else "geleverd"),
        "price": f"€ {index % 300 + 20},95",
        "availability": ("op voorraad" if index % 2 else "tijdelijk uitverkocht"),
        "author": f"{last}, {first[0]}.",
        "title": f"Onderzoek naar taalmodellen deel {index % 20 + 1}",
        "year": str(2020 + index % 7),
        "address": f"Voorbeeldstraat {index % 250 + 1}",
        "postal_code": f"{1000 + index % 8000} AB",
        "sender": f"team{index}@voorbeeld.nl",
        "subject": f"Update dossier nummer {index}",
        "message": f"De afspraak is bevestigd voor dossier {index}.",
        "event": f"Data-avond sessie {index % 50 + 1}",
        "venue": ("De Vooruit" if index % 2 else "Jaarbeurs"),
        "merchant": f"Winkel Voorbeeld {index % 30 + 1}",
        "currency": ("EUR" if index % 3 else "GBP"),
        "category": ("reizen" if index % 2 else "kantoor"),
        "recipient": f"Ontvanger {first} {last}",
        "item": ("usb-c-kabel" if index % 2 else "notitieboek"),
        "unit_price": f"€ {index % 80 + 5},50",
        "doctor": doctors[index % len(doctors)],
        "specialty": specialties[index % len(specialties)],
        "iban": f"NL{10 + index % 80:02d}BANK0{index % 900000000 + 100000000:09d}",
        "amount": f"€ {index % 450 + 25},50",
        "description": f"Factuur referentie {index}",
        "hotel": hotels[index % len(hotels)],
        "city": city,
        "checkin": f"2026-11-{index % 25 + 1:02d}",
        "nights": str(index % 7 + 1),
        "airline": airlines[index % len(airlines)],
        "flight_number": f"KL{index % 8000 + 1000:04d}",
        "origin": city,
        "destination": dest_city,
        "customer": full_name,
        "issue": issues[index % len(issues)],
        "priority": ("hoog" if index % 2 else "normaal"),
        "order_id": f"ORD-{index % 900000 + 100000:06d}",
        "reason": reasons[index % len(reasons)],
    }
    return {field: values[field] for field in fields}


def _extraction_query(values: dict[str, str], locale: str, index: int = 0) -> str:
    pairs = ". ".join(f"{field.replace('_', ' ').capitalize()}: {value}" for field, value in values.items())
    if locale == "nl-BE":
        openers = [
            "Kunt ge dit even verwerken? ",
            "Wilt ge de volgende gegevens registreren: ",
            "Gelieve deze velden over te nemen: ",
        ]
        return openers[index % len(openers)] + pairs + "."
    if locale == "nl-noisy":
        return "pls haal dit er uit: " + pairs.lower() + "."
    openers = [
        "Haal de gegevens uit deze tekst: ",
        "Zet de volgende informatie om in gestructureerde velden: ",
        "Lees het onderstaande overzicht en registreer de waarden: ",
        "Verwerk deze gegevens nauwkeurig: ",
    ]
    return openers[index % len(openers)] + pairs + "."


def make_extraction_examples(count: int, *, split: str, seed: int = 0,
                             start: int = 0) -> list[dict[str, Any]]:
    """Create labelled, evidence-preserving Dutch extraction examples."""
    rows = []
    for offset in range(count):
        index = start + offset + seed * 10_000
        name, fields = _EXTRACTION_SCHEMAS[index % len(_EXTRACTION_SCHEMAS)]
        locale = _dialect(index)
        schema_variant = 2 if split == "test" else index % 2
        values = _extraction_values(name, fields, index)
        tools = _candidate_extraction_tools(index, variant=schema_variant)
        # Do not let the target tool's catalogue position become a shortcut.
        # Keep the order reproducible so regenerated datasets are identical.
        random.Random(index * 104729 + schema_variant).shuffle(tools)
        query = _extraction_query(values, locale, index) + f" Referentie {index}."
        rows.append({
            "id": f"extraction:{split}:{index}", "group_id": f"extraction:{index}",
            "source": "deterministic-dutch-extraction-v1", "locale": locale,
            "domain": "extraction", "extraction_type": name,
            "schema_variant": schema_variant, "tool_similarity": "same_record_shape",
            "task_family": "extraction", "split": split, "query": query,
            "tools": tools, "answers": [{"name": name, "arguments": values}],
            "reasoning": "; ".join(f"{field}='{value}' uit de tekst"
                                    for field, value in values.items()),
        })
    return rows


def make_negative_examples(count: int, *, split: str, seed: int = 0,
                           start: int = 0) -> list[dict[str, Any]]:
    topics = [
        "Wat is de hoofdstad van IJsland, volgens vraag {index}?",
        "Vertel een kort verhaal over vos {index}.",
        "Waarom is de lucht blauw bij voorbeeld {index}?",
        "Ik wil geen alarm om {time} instellen, ik denk alleen hardop.",
        "Kun je deze zin grammaticaal uitleggen, versie {index}?",
        "Welke film raad je aan voor avond {index}?",
        "Het pakket {tracking} is al geleverd, verander niets.",
        "Kun je uitleggen wat {minutes} minuten betekent?",
        "Wat zijn de voordelen van zonne-energie in scenario {index}?",
        "Hoeveel kilometer is het ongeveer fietsen van Amsterdam naar Utrecht (vraag {index})?",
        "Ik overweeg om om {time} op te staan, maar activeer nog geen wekker.",
        "Laat de instellingen ongewijzigd voor bestelling {index}, ik kijk alleen.",
        "Wat is het verschil tussen een virus en een bacterie bij toets {index}?",
        "Dankjewel voor het overzicht van dossier {index}, fijne dag verder!",
        "Stel géén timer in van {minutes} minuten, dat was een vergissing.",
        "Welke boeken over geschiedenis raad je aan voor thema {index}?",
    ]
    action_tools = [
        {"name": "set_alarm", "description": "Set an alarm.", "parameters": {
            "type": "object", "properties": {"time": {"type": "string"}}, "required": ["time"]}},
        {"name": "set_timer", "description": "Set a timer.", "parameters": {
            "type": "object", "properties": {"minutes": {"type": "integer"}}, "required": ["minutes"]}},
        {"name": "track_shipment", "description": "Track a shipment.", "parameters": {
            "type": "object", "properties": {"tracking_number": {"type": "string"}}, "required": ["tracking_number"]}},
    ]
    rows = []
    for offset in range(count):
        index = start + offset + seed * 10_000
        query = topics[index % len(topics)].format(
            index=index, time=f"{6 + index % 15:02d}:30", minutes=index % 50 + 1,
            tracking=f"3SNL{index % 90000000000 + 10000000000:011d}")
        query += f" Referentie {index}."
        tools = action_tools if index % 2 else _candidate_extraction_tools(index, variant=index % 2)
        rows.append({
            "id": f"negative:{split}:{index}", "group_id": f"negative:{index}",
            "source": "deterministic-dutch-negative-v1", "locale": _dialect(index),
            "domain": "negative", "schema_variant": 2 if split == "test" else index % 2,
            "tool_similarity": "unrelated", "task_family": "negative", "split": split,
            "query": query, "tools": tools, "answers": [],
        })
    return rows


def make_multi_call_examples(count: int, *, split: str, seed: int = 0,
                             start: int = 0) -> list[dict[str, Any]]:
    rows = []
    for offset in range(count):
        index = start + offset + seed * 10_000
        scenario = index % 4
        schema_variant = 2 if split == "test" else index % 2
        locale = _dialect(index)

        if scenario == 0:
            tools = [
                {"name": "set_alarm", "description": "Set an alarm.", "parameters": {
                    "type": "object", "properties": {"time": {"type": "string"}}, "required": ["time"]}},
                {"name": "set_timer", "description": "Set a timer.", "parameters": {
                    "type": "object", "properties": {"minutes": {"type": "integer", "minimum": 1}}, "required": ["minutes"]}},
                {"name": "get_weather", "description": "Get current weather.", "parameters": {
                    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
            ]
            hour, minutes = 6 + index % 15, 5 + index % 55
            timer = index % 50 + 1
            queries = [
                "Zet een alarm om {time} en een timer van {timer} minuten.",
                "Wil je een timer van {timer} minuten starten en mij om {time} wekken?",
                "Kunt ge zowel een alarm voor {time} als een timer van {timer} minuten zetten?",
            ]
            query = queries[index % len(queries)].format(
                time=f"{hour:02d}:{minutes:02d}", timer=timer) + f" Referentie {index}."
            answers = [
                {"name": "set_alarm", "arguments": {"time": f"{hour:02d}:{minutes:02d}"}},
                {"name": "set_timer", "arguments": {"minutes": timer}},
            ]
            # Match answer order to the request order for the timer-first wording.
            if index % len(queries) == 1:
                answers.reverse()
            reasoning = f"time='{hour:02d}:{minutes:02d}' en minutes='{timer}' uit de vraag"
            domain = "time"
        elif scenario == 1:
            tools = [
                {"name": "control_lights", "description": "Control lights.", "parameters": {
                    "type": "object", "properties": {"room": {"type": "string"}, "state": {"type": "string"}}, "required": ["room", "state"]}},
                {"name": "set_temperature", "description": "Set room temperature.", "parameters": {
                    "type": "object", "properties": {"room": {"type": "string"}, "temperature": {"type": "string"}}, "required": ["room", "temperature"]}},
                {"name": "lock_door", "description": "Lock a door.", "parameters": {
                    "type": "object", "properties": {"door": {"type": "string"}, "status": {"type": "string"}}, "required": ["door", "status"]}},
            ]
            rooms = ["woonkamer", "keuken", "slaapkamer", "kantoor", "badkamer"]
            room = rooms[index % len(rooms)]
            temp = f"{18 + index % 5} °C"
            state = "aan" if index % 2 == 0 else "uit"
            queries = [
                "Doe het licht in de {room} {state} en zet de temperatuur op {temp}.",
                "Schakel de lampen in de {room} {state} en regel de thermostaat van de {room} naar {temp}.",
                "Kunt ge in de {room} de verlichting {state} zetten en de temperatuur instellen op {temp}?",
            ]
            query = queries[index % len(queries)].format(room=room, state=state, temp=temp) + f" Referentie {index}."
            answers = [
                {"name": "control_lights", "arguments": {"room": room, "state": state}},
                {"name": "set_temperature", "arguments": {"room": room, "temperature": temp}},
            ]
            reasoning = f"room='{room}', state='{state}' en temperature='{temp}' uit de vraag"
            domain = "home"
        elif scenario == 2:
            tools = [
                {"name": "create_appointment", "description": "Schedule appointment.", "parameters": {
                    "type": "object", "properties": {"date": {"type": "string"}, "time": {"type": "string"}, "location": {"type": "string"}}, "required": ["date", "time", "location"]}},
                {"name": "send_message", "description": "Send a message.", "parameters": {
                    "type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}}, "required": ["recipient", "message"]}},
                {"name": "find_contact", "description": "Find contact.", "parameters": {
                    "type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}}, "required": ["name", "email"]}},
            ]
            date_val = f"2026-11-{index % 25 + 1:02d}"
            time_val = f"{9 + index % 8:02d}:00"
            loc_val = "Gent" if index % 2 == 0 else "Utrecht"
            recipient = f"collega{index}@voorbeeld.nl"
            msg = f"afspraak gepland op {date_val}"
            queries = [
                "Plan een afspraak op {date} om {time} in {location} en stuur naar {recipient}: {message}.",
                "Zet een ontmoeting vast op {date} om {time} bij {location} en laat {recipient} weten dat {message}.",
            ]
            query = queries[index % len(queries)].format(
                date=date_val, time=time_val, location=loc_val, recipient=recipient, message=msg) + f" Referentie {index}."
            answers = [
                {"name": "create_appointment", "arguments": {"date": date_val, "time": time_val, "location": loc_val}},
                {"name": "send_message", "arguments": {"recipient": recipient, "message": msg}},
            ]
            reasoning = f"date='{date_val}', time='{time_val}', location='{loc_val}', recipient='{recipient}' uit de vraag"
            domain = "scheduling"
        else:
            tools = [
                {"name": "create_note", "description": "Create a note.", "parameters": {
                    "type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]}},
                {"name": "add_todo_item", "description": "Add todo item.", "parameters": {
                    "type": "object", "properties": {"task": {"type": "string"}, "priority": {"type": "string"}}, "required": ["task", "priority"]}},
                {"name": "send_message", "description": "Send a message.", "parameters": {
                    "type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}}, "required": ["recipient", "message"]}},
            ]
            title = f"Projectnotitie {index}"
            content = f"Actiepunten voor sprint {index % 10 + 1}"
            task = f"kwartaaloverleg voorbereiden {index}"
            priority = "hoog" if index % 2 == 0 else "gemiddeld"
            queries = [
                "Maak een notitie getiteld '{title}' met inhoud '{content}' en voeg de taak '{task}' toe met prioriteit {priority}.",
                "Noteer '{content}' onder titel '{title}' en zet tevens '{task}' op mijn takenlijst met status {priority}.",
            ]
            query = queries[index % len(queries)].format(
                title=title, content=content, task=task, priority=priority) + f" Referentie {index}."
            answers = [
                {"name": "create_note", "arguments": {"title": title, "content": content}},
                {"name": "add_todo_item", "arguments": {"task": task, "priority": priority}},
            ]
            reasoning = f"title='{title}', content='{content}', task='{task}' en priority='{priority}' uit de vraag"
            domain = "productivity"

        # Every scenario has a fresh list, so shuffling here is safe and removes
        # a systematic "correct tool is first" signal from deterministic data.
        random.Random(index * 104729 + scenario * 1009 + schema_variant).shuffle(tools)
        rows.append({
            "id": f"multi:{split}:{index}", "group_id": f"multi:{index}",
            "source": "deterministic-dutch-multi-v1", "locale": locale,
            "domain": domain, "schema_variant": schema_variant,
            "tool_similarity": "mixed_workflow_actions", "task_family": "multi_call", "split": split,
            "query": query, "tools": tools, "answers": answers, "reasoning": reasoning,
        })
    return rows


# A comprehensive deterministic vocabulary and template bank for Dutch function calling.
# All values are placed verbatim in generated queries to ensure 100% valid grounding.
_AUGMENTED_TOOL_SPECS: dict[str, dict[str, str]] = {
    "set_alarm": {"time": "string"},
    "set_timer": {"minutes": "integer"},
    "get_weather": {"city": "string"},
    "track_shipment": {"tracking_number": "string", "carrier": "string"},
    "create_appointment": {"date": "string", "time": "string", "location": "string"},
    "send_message": {"recipient": "string", "message": "string"},
    "find_contact": {"name": "string", "email": "string"},
    "search_product": {"product": "string", "category": "string"},
    "update_order": {"order_number": "string", "status": "string"},
    "create_event": {"event": "string", "date": "string", "venue": "string"},
    "move_robot": {"destination": "string", "speed": "string"},
    "pick_up_object": {"object": "string", "location": "string"},
    "inspect_object": {"object": "string"},
    "set_robot_mode": {"mode": "string"},
    "analyze_sentiment": {"text": "string"},
    "launch_game": {"game": "string", "platform": "string"},
    "invite_player": {"player": "string", "game": "string"},
    "set_game_difficulty": {"game": "string", "difficulty": "string"},
    "report_player": {"player": "string", "reason": "string"},
    "control_lights": {"room": "string", "state": "string"},
    "set_temperature": {"room": "string", "temperature": "string"},
    "book_ticket": {"destination": "string", "date": "string", "passenger": "string"},
    "play_music": {"artist": "string", "genre": "string"},
    "translate_text": {"text": "string", "target_language": "string"},
    "send_email": {"recipient": "string", "subject": "string", "body": "string"},
    "create_note": {"title": "string", "content": "string"},
    "add_todo_item": {"task": "string", "priority": "string"},
    "delete_alarm": {"time": "string"},
    "cancel_appointment": {"date": "string", "contact": "string"},
    "reschedule_meeting": {"date": "string", "new_time": "string", "person": "string"},
    "transfer_money": {"recipient": "string", "amount": "string", "description": "string"},
    "check_balance": {"account": "string"},
    "pay_invoice": {"invoice_number": "string", "amount": "string", "vendor": "string"},
    "lock_door": {"door": "string", "status": "string"},
    "set_fan_speed": {"room": "string", "fan_speed": "string"},
    "open_curtains": {"room": "string", "percentage": "string"},
    "book_hotel": {"hotel": "string", "city": "string", "date": "string"},
    "check_flight_status": {"flight_number": "string", "airline": "string"},
    "add_to_cart": {"product": "string", "quantity": "integer"},
    "cancel_order": {"order_number": "string", "reason": "string"},
    "request_refund": {"order_number": "string", "amount": "string"},
    "check_stock": {"product": "string", "store": "string"},
    "pause_playback": {"device": "string"},
    "adjust_volume": {"device": "string", "level": "string"},
    "create_playlist": {"playlist_name": "string", "genre": "string"},
    "schedule_doctor_visit": {"doctor": "string", "date": "string", "time": "string"},
    "log_symptoms": {"symptom": "string", "severity": "string"},
    "find_route": {"origin": "string", "destination": "string"},
    "summarize_document": {"title": "string"},
    "toggle_device": {"device_name": "string", "state": "string"},
    "archive_email": {"subject": "string", "sender": "string"},
    "set_reminder": {"reminder": "string", "time": "string"},
}

_AUGMENTED_LEXICON = {
    "cities": [
        "Amsterdam", "Antwerpen", "Arnhem", "Brugge", "Brussel", "Delft",
        "Eindhoven", "Gent", "Groningen", "Haarlem", "Leeuwarden", "Leiden",
        "Leuven", "Maastricht", "Mechelen", "Nijmegen", "Rotterdam", "Utrecht",
        "Venlo", "Vlissingen", "Zwolle", "Aalst", "Hasselt", "Kortrijk",
        "Amersfoort", "Apeldoorn", "Dordrecht", "Enschede", "Oostende",
        "Sint-Niklaas", "Turnhout", "Genk", "Roeselare", "Almere", "Breda",
        "Tilburg", "Den Haag", "'s-Hertogenbosch", "Hilversum", "Gouda",
    ],
    "names": [
        "Amina El Idrissi", "Bram Jansen", "Charlotte de Wit", "Daan Peeters",
        "Elise Smit", "Fien Vermeulen", "Gijs van Dijk", "Hanne Jacobs",
        "Ilias Bakker", "Joris de Vries", "Kim Willems", "Lotte Maes",
        "Milan Visser", "Nora Hendriks", "Olivier Claes", "Puck de Boer",
        "Ruben Van den Bossche", "Sanne Wouters", "Thijs Hermans", "Yara Vos",
        "Willem Aarts", "Emma Meijer", "Lars Brouwer", "Anouk de Jong",
        "Sofie Van Damme", "Koen Verhoeven", "Fatima Bouzian", "Jasper Smet",
        "Liesbeth Timmermans", "Sander Van de Velde", "Tess Mulder", "Stijn Goossens",
    ],
    "products": [
        "draadloze koptelefoon", "koffiemolen", "e-reader", "usb-c-kabel",
        "regenjas", "wandelschoenen", "staande lamp", "steelpan",
        "notitieboek", "fietsslot", "webcam", "bluetoothspeaker",
        "powerbank", "thermosfles", "bureauventilator", "slaapmasker",
        "laptoptas", "draadloze muis", "slimme stekker", "waterkoker",
        "sporthorloge", "fietspomp", "keukenweegschaal", "bureaustoel",
        "toetsenbord", "rugzak", "ledstrip", "yogamat", "strijkijzer",
    ],
    "categories": [
        "elektronica", "keuken", "kleding", "boeken", "fietsen", "kantoor",
        "wonen", "reizen", "sport", "cadeaus", "tuin", "verzorging", "gereedschap",
        "speelgoed", "huisdieren", "audio", "verlichting", "huishouden",
    ],
    "carriers": ["PostNL", "bpost", "DHL", "DPD", "GLS", "UPS", "Mondial Relay", "FedEx"],
    "statuses": [
        "onderweg", "geleverd", "vertraagd", "geannuleerd", "klaar voor afhaling",
        "in behandeling", "teruggestuurd", "vergrendeld", "ontgrendeld",
    ],
    "locations": [
        "Stationsplein 12, Gent", "Damrak 1, Amsterdam", "Korenmarkt 5, Gent",
        "Meir 78, Antwerpen", "Neude 11, Utrecht", "Rue de la Loi 16, Brussel",
        "Witte de Withstraat 30, Rotterdam", "Oude Markt 9, Leuven",
        "Grote Markt 1, Brugge", "Vrijthof 4, Maastricht", "Grote Markt 20, Groningen",
        "Coolsingel 40, Rotterdam", "Janskerkhof 3, Utrecht", "De Keyserlei 15, Antwerpen",
    ],
    "recipients": [
        "planning@voorbeeld.nl", "team@voorbeeld.be", "oma@voorbeeld.nl",
        "boekhouding@voorbeeld.be", "samira@voorbeeld.nl", "receptie@voorbeeld.be",
        "klantenservice@voorbeeld.nl", "directie@voorbeeld.be", "support@voorbeeld.nl",
        "info@voorbeeld.be", "hr@voorbeeld.nl", "logistiek@voorbeeld.be",
    ],
    "messages": [
        "de afspraak is bevestigd", "ik ben tien minuten later", "de offerte staat klaar",
        "kun je dit vandaag nakijken", "de bestelling mag worden verzonden",
        "bedankt voor de snelle reactie", "de vergadering is verplaatst",
        "het dossier is volledig bijgewerkt", "we zien elkaar morgenochtend",
        "kun je de factuur nog even controleren", "het pakket is zojuist aangekomen",
        "ik stuur de bijlage zo dadelijk door", "de presentatie is goedgekeurd",
    ],
    "events": [
        "Workshop digitale toegankelijkheid", "Buurtvergadering", "Boekenclub",
        "Herfstconcert", "Teamuitje", "Lezing over geschiedenis", "Kookavond",
        "Kwartaaloverleg", "Jaarlijkse Algemene Ledenvergadering", "Tech Meetup",
        "Ouderavond", "Netwerkborrel", "Creatieve Schrijfsessie", "Productlancering",
    ],
    "venues": [
        "De Vooruit", "Jaarbeurs", "Bozar", "Bibliotheek Permeke", "TivoliVredenburg",
        "Concertgebouw Brugge", "Muziekgebouw aan 't IJ", "De Roma", "Het Depot Leuven",
        "Kinepolis Antwerpen", "Westergas Amsterdam", "Stadsschouwburg Utrecht",
    ],
    "authors": [
        "Bantilan, N.", "Jansen, M.", "De Smet, L.", "Van den Berg, P.",
        "Willems, K.", "Claes, O.", "Vermeulen, F.", "Hendriks, N.",
    ],
    "titles": [
        "Onderzoek naar taalmodellen", "Praktische gids voor privacy",
        "De geschiedenis van de Lage Landen", "Inleiding tot informatieontwerp",
        "Duurzame technologie in de praktijk", "Moderne software-architectuur",
        "Gezond leven in de stad", "Leren programmeren met plezier",
    ],
    "robot_destinations": [
        "de keuken", "de woonkamer", "magazijnvak B3", "de laadzone",
        "de voordeur", "het laboratorium", "de vergaderruimte", "het terras",
        "de lift", "de werkbank", "gang C", "de ontvangstbalie",
        "de serverruimte", "de archiefruimte", "de kantine", "de opslagloods",
    ],
    "robot_objects": [
        "de rode mok", "het pakket", "de blauwe doos", "de schroevendraaier",
        "de handdoek", "het boek", "de afstandsbediening", "de sleutelbos",
        "de kleine robot", "het glas", "de plantenpot", "de gele bal",
        "het meetlint", "de veiligheidsbril", "het dossier", "de waterfles",
    ],
    "robot_speeds": ["langzaam", "normaal", "snel", "extra voorzichtig"],
    "robot_modes": ["patrouille", "oplaadmodus", "stil", "handmatig", "standby"],
    "robot_rooms": [
        "keuken", "woonkamer", "slaapkamer", "hal", "kantoor", "garage",
        "badkamer", "zolder", "berging", "studeerkamer", "eetkamer",
    ],
    "robot_states": ["aan", "uit", "gedimd", "helder"],
    "temperatures": ["17 °C", "18 °C", "19 °C", "20 °C", "21 °C", "22 °C", "23 °C"],
    "games": [
        "Minecraft", "Rocket League", "Stardew Valley", "Fortnite", "Hades",
        "Mario Kart", "Civilization VI", "The Sims", "Baldur's Gate 3", "Overwatch",
        "EA Sports FC", "Zelda: Tears of the Kingdom", "Cyberpunk 2077", "Valorant",
    ],
    "platforms": ["pc", "PlayStation 5", "Xbox Series X", "Nintendo Switch", "Steam Deck", "mobiel"],
    "players": ["Ruben42", "LotteGaming", "NoobMetKoffie", "Ayla_07", "PixelPiet", "SofiePro", "DutchGamer99", "VlaamseLeeuw"],
    "difficulties": ["makkelijk", "normaal", "moeilijk", "expert"],
    "report_reasons": ["schelden", "valsspelen", "ongewenste berichten", "bedrog", "spam", "onsportief gedrag"],
    "genres": [
        "jazz", "klassieke muziek", "indiepop", "elektronische muziek", "folk",
        "rock", "hiphop", "nederpop", "ambient", "blues", "techno",
    ],
    "artists": [
        "Spinvis", "Angèle", "Boudewijn de Groot", "Racoon", "Stromae", "Froukje",
        "Eefje de Visser", "Goldband", "dEUS", "De Jeugd van Tegenwoordig", "S10", "Tourist LeMC",
    ],
    "languages": ["Engels", "Frans", "Duits", "Spaans", "Italiaans", "Portugees", "Pools", "Arabisch", "Japans"],
    "passengers": [
        "Amina El Idrissi", "Bram Jansen", "Charlotte de Wit", "Daan Peeters",
        "Elise Smit", "Fien Vermeulen", "Gijs van Dijk", "Hanne Jacobs",
    ],
    "currencies": ["EUR", "GBP", "USD", "CHF"],
    "expense_categories": [
        "reizen", "kantoor", "maaltijden", "materiaal", "opleiding",
        "software", "marketing", "vervoer", "huisvesting", "telecom",
    ],
    "doctors": [
        "Dr. Peeters", "Dr. Van Dijk", "Dr. De Smet", "Dr. Bakker",
        "Dr. Visser", "Dr. Willems", "Dr. Maes", "Dr. Jansen",
    ],
    "specialties": [
        "huisarts", "tandarts", "oogarts", "dermatoloog",
        "fysiotherapeut", "cardioloog", "neuroloog", "psycholoog",
    ],
    "hotels": [
        "Hotel De Korenbeurs", "Grand Hotel Karel V", "The Dominican",
        "Boutique Hotel Gravensteen", "Hotel Vondelpark", "Pillows Grand Boutique Hotel",
    ],
    "airlines": ["KLM", "Brussels Airlines", "Transavia", "Lufthansa", "Air France", "EasyJet"],
    "flight_numbers": ["KL1234", "SN3721", "HV6112", "LH2304", "AF1456", "U27891"],
    "amounts": ["€ 15,50", "€ 42,00", "€ 75,25", "€ 120,00", "€ 250,00", "€ 450,00", "€ 89,95", "€ 19,99", "€ 340,50"],
    "accounts": ["betaalrekening", "spaarrekening", "zakelijke rekening", "gezamenlijke rekening"],
    "descriptions": [
        "contributie sportclub", "lunch met klant", "huur november",
        "factuur boekhouding", "verjaardagscadeau", "energiekosten", "boodschappen",
    ],
    "doors": ["voordeur", "achterdeur", "garagedeur", "schuifpui", "balkondeur"],
    "devices": ["soundbar", "slimme speaker", "televisie", "versterker", "radio"],
    "fan_speeds": ["stand 1", "stand 2", "stand 3", "automatisch", "turbo", "stil"],
    "curtain_percentages": ["25%", "50%", "75%", "100%", "0%"],
    "priorities": ["laag", "gemiddeld", "hoog", "urgent"],
    "severities": ["mild", "matig", "ernstig"],
    "symptoms": ["hoofdpijn", "keelpijn", "vermoeidheid", "koorts", "spierpijn", "rugpijn", "hoesten"],
    "tasks": [
        "presentatie voorbereiden", "kwartaalrapport afronden", "facturen controleren",
        "teamvergadering plannen", "e-mails beantwoorden", "back-up maken", "contract nalezen",
    ],
    "note_titles": [
        "Notities projectoverleg", "Ideeën voor marketing", "Boodschappenlijst weekend",
        "Checklist verhuizing", "Samenvatting lezing", "Planning kwartaal 4",
    ],
    "reasons": [
        "verkeerde maat", "beschadigd artikel", "defect bij levering",
        "niet naar verwachting", "dubbel besteld", "te laat bezorgd",
    ],
    "stores": ["Filiaal Centrum", "Winkel West", "Megastore Noord", "Online Magazijn"],
    "volume_levels": ["laag", "gemiddeld", "hoog", "dempen", "maximaal"],
    "playlist_names": ["Focus & Werk", "Weekend Vibes", "Ontspanning", "Hardloopbeats", "Avondrust"],
    "email_subjects": [
        "Belangrijke update", "Vraag over offerte", "Bevestiging afspraak",
        "Notulen vergadering", "Voortgangsproject", "Factuuroverzicht",
    ],
    "reminders": [
        "medicijnen innemen", "planten water geven", "was ophangen",
        "vuilnis buiten zetten", "treinkaartje kopen", "terugbellen naar kantoor",
    ],
}

_AUGMENTED_SENTIMENTS = [
    ("De klantenservice reageerde snel en vriendelijk.", "positief"),
    ("De applicatie crasht telkens na het inloggen.", "negatief"),
    ("De bestelling is vandaag om twaalf uur aangekomen.", "neutraal"),
    ("Wat een fijne en overzichtelijke website.", "positief"),
    ("Ik ben teleurgesteld door de lange wachttijd.", "negatief"),
    ("Het abonnement kost twintig euro per maand.", "neutraal"),
    ("Fantastische service en snelle levering van mijn bestelling.", "positief"),
    ("Het product vertoont al na twee dagen duidelijke gebreken.", "negatief"),
    ("De winkel opent elke werkdag om negen uur.", "neutraal"),
    ("Geweldige gebruikerservaring en duidelijke interface.", "positief"),
    ("De verbinding valt voortdurend weg tijdens het bellen.", "negatief"),
    ("De bijeenkomst vindt plaats in zaal drie.", "neutraal"),
]

_AUGMENTED_ENUMS = {
    "sentiment": ["positief", "negatief", "neutraal"],
    "speed": ["langzaam", "normaal", "snel", "extra voorzichtig"],
    "fan_speed": ["stand 1", "stand 2", "stand 3", "automatisch", "turbo", "stil"],
    "state": ["aan", "uit", "gedimd", "helder"],
    "difficulty": ["makkelijk", "normaal", "moeilijk", "expert"],
    "mode": ["patrouille", "oplaadmodus", "stil", "handmatig", "standby"],
    "priority": ["laag", "gemiddeld", "hoog", "urgent"],
    "severity": ["mild", "matig", "ernstig"],
    "status": [
        "onderweg", "geleverd", "vertraagd", "geannuleerd", "klaar voor afhaling",
        "in behandeling", "teruggestuurd", "vergrendeld", "ontgrendeld",
    ],
}


def _augmented_locale(index: int) -> str:
    # Keep dialect assignment independent from the task-family bucket in
    # ``make_augmented_examples``.  Using ``index % 10`` for both would teach
    # the model that actions are NL-NL, extraction is NL-BE, and multi-call
    # requests are noisy Dutch instead of learning the task itself.
    remainder = (index // 10) % 10
    if remainder < 7:
        return "nl-NL"
    if remainder < 9:
        return "nl-BE"
    return "nl-noisy"


def _augmented_value(key: str, index: int) -> str:
    values = _AUGMENTED_LEXICON[key]
    return values[index % len(values)]


def _augmented_date(index: int) -> str:
    return f"2027-{index % 12 + 1:02d}-{index % 27 + 1:02d}"


def _augmented_time(index: int) -> str:
    return f"{6 + index % 15:02d}:{(index * 7) % 60:02d}"


_AUGMENTED_CONTEXTS = [
    "", "Alsjeblieft: ", "Voor mijn planning: ", "Graag: ",
    "Even tussendoor: ", "Voor vandaag: ", "Kun je dit regelen? ",
    "Ik wil het volgende: ", "Belangrijk: ", "Zo snel mogelijk: ",
    "Als het even kan: ", "Met spoed: ", "Voor het project: ",
    "Ter herinnering: ", "Gelieve het volgende te verwerken: ",
    "Vraagje tussendoor: ", "Noteer even: ", "Kun je me helpen met: ",
    "Kijk hier eens naar: ", "Doe me een plezier en: ",
]


def _augmented_context(query: str, index: int) -> str:
    # A prime stride keeps the register variation independent from the
    # vocabulary/template cycles, avoiding a small repeating query set.
    return _AUGMENTED_CONTEXTS[(index // 37) % len(_AUGMENTED_CONTEXTS)] + query


def _augmented_typo(query: str, values: dict[str, Any]) -> str:
    """Add common typos only to surrounding prose, never to evidence values."""
    folded_values = [str(value).casefold() for value in values.values()]
    replacements = [
        ("alsjeblieft", "alstublieft"), ("instellen", "instelle"),
        ("verwerken", "verwerkn"), ("bestelling", "besteling"),
        ("weersverwachting", "weersverwachtingg"), ("informatie", "informtie"),
        ("controleer", "kontroleer"), ("afspraak", "afspraakk"),
        ("temperatuur", "tempratuur"), ("vergadering", "vergadring"),
        ("herinnering", "herinering"), ("bevestigen", "bevestign"),
    ]
    for source, target in replacements:
        if any(source in value for value in folded_values):
            continue
        query = query.replace(source, target).replace(source.capitalize(), target.capitalize())
    return query


def _augmented_tool(name: str, fields: dict[str, str], variant: int) -> dict[str, Any]:
    descriptions = [
        "Use this tool for the requested operation.",
        "Perform the user's requested action and return its result.",
        "Handle this structured request using the supplied fields.",
    ]
    return {
        "name": name,
        "description": descriptions[variant % len(descriptions)],
        "parameters": {
            "type": "object",
            "properties": {
                field: {
                    "type": field_type,
                    "description": field.replace("_", " "),
                    **({"enum": _AUGMENTED_ENUMS[field]} if field in _AUGMENTED_ENUMS else {}),
                }
                for field, field_type in fields.items()
            },
            "required": list(fields),
        },
    }


def _augmented_catalogue(target: str, index: int, variant: int) -> list[dict[str, Any]]:
    names = list(_AUGMENTED_TOOL_SPECS)
    rng = random.Random(index * 7919 + variant)
    distractors = [name for name in names if name != target]
    rng.shuffle(distractors)
    selected = [target] + distractors[: 2 + index % 3]
    rng.shuffle(selected)
    return [_augmented_tool(name, _AUGMENTED_TOOL_SPECS[name], variant) for name in selected]


def _augmented_action(index: int, split: str) -> dict[str, Any]:
    """Create one varied Dutch action or extraction record."""
    kind = (index // 10) % 52
    locale = _augmented_locale(index)
    variant = 2 if split == "test" else index % 2

    if kind == 0:
        target, values = "set_alarm", {"time": _augmented_time(index)}
        templates = [
            "Zet een alarm om {time}.", "Kun je me om {time} wakker maken?",
            "Graag een wekker instellen voor {time}.", "Ik wil dat je om {time} een alarm zet.",
            "Herinner me eraan om {time} op te staan.", "Stel voor {time} een ochtendalarm in.",
            "Om {time} moet mijn alarm afgaan.", "Maak een wekker met tijd {time}.",
        ]
    elif kind == 1:
        target, values = "set_timer", {"minutes": 5 + index % 176}
        templates = [
            "Start een timer van {minutes} minuten.", "Tel {minutes} minuten af.",
            "Zet een kookwekker op {minutes} minuten.", "Ik heb {minutes} minuten nodig; start de timer.",
            "Laat {minutes} minuten lopen op de timer.", "Kun je {minutes} minuten timen?",
            "Begin een aftelling van {minutes} minuten.", "Stel de timer in op {minutes} minuten.",
        ]
    elif kind == 2:
        target, values = "get_weather", {"city": _augmented_value("cities", index)}
        templates = [
            "Hoe wordt het weer in {city}?", "Geef me de weersverwachting voor {city}.",
            "Regent het vandaag in {city}?", "Wat zijn de temperaturen in {city}?",
            "Is een jas nodig in {city} vandaag?", "Toon de actuele weersinformatie voor {city}.",
            "Wat voorspelt het weerbericht voor {city}?", "Hoe warm wordt het straks in {city}?",
        ]
    elif kind == 3:
        target, values = "track_shipment", {
            "tracking_number": f"3SNL{index % 90000000000 + 10000000000:011d}",
            "carrier": _augmented_value("carriers", index),
        }
        templates = [
            "Volg pakket {tracking_number} met {carrier}.",
            "Kun je de zending {tracking_number} bij {carrier} volgen?",
            "Waar is mijn {carrier}-pakket met code {tracking_number}?",
            "Controleer bij {carrier} de status van {tracking_number}.",
            "Zoek de locatie van zending {tracking_number} ({carrier}).",
            "Geef een update over {tracking_number}, vervoerder {carrier}.",
        ]
    elif kind == 4:
        target, values = "create_appointment", {
            "date": _augmented_date(index), "time": _augmented_time(index),
            "location": _augmented_value("locations", index),
        }
        templates = [
            "Plan een afspraak op {date} om {time} op {location}.",
            "Zet {date} om {time} vast bij {location}.",
            "Ik wil naar {location} op {date} om {time}; maak een afspraak.",
            "Boek een afspraak voor {date}, {time}, locatie {location}.",
            "Reserveer {time} op {date} bij {location}.",
            "Maak een afspraak in {location} voor {date} om {time}.",
        ]
    elif kind == 5:
        target, values = "send_message", {
            "recipient": _augmented_value("recipients", index),
            "message": _augmented_value("messages", index),
        }
        templates = [
            "Stuur naar {recipient}: {message}.",
            "Maak een bericht voor {recipient} met de tekst '{message}'.",
            "Sms {recipient} dat {message}.",
            "Verstuur aan {recipient} het volgende: {message}.",
            "Laat {recipient} weten dat {message}.",
            "Schrijf voor {recipient}: {message}.",
        ]
    elif kind == 6:
        target, values = "find_contact", {
            "name": _augmented_value("names", index),
            "email": f"contact{index % 9000 + 1000}@voorbeeld.nl",
        }
        templates = [
            "Zoek het contact {name}, e-mail {email}.",
            "Vind {name}; het e-mailadres is {email}.",
            "Toon de contactgegevens van {name} met {email}.",
            "Open het adresboek en zoek {name} ({email}).",
            "Kun je {name} vinden? Gebruik e-mail {email}.",
            "Zoek {name} in mijn contacten; adres {email}.",
        ]
    elif kind == 7:
        target, values = "search_product", {
            "product": _augmented_value("products", index),
            "category": _augmented_value("categories", index),
        }
        templates = [
            "Zoek een {product} in de categorie {category}.",
            "Vind {product}; filter op {category}.",
            "Ik zoek binnen {category} naar een {product}.",
            "Toon producten zoals {product} onder {category}.",
            "Doorzoek {category} voor {product}.",
            "Waar vind ik een {product} in {category}?",
        ]
    elif kind == 8:
        target, values = "update_order", {
            "order_number": f"ORD-{index % 900000 + 100000:07d}",
            "status": _augmented_value("statuses", index),
        }
        templates = [
            "Werk bestelling {order_number} bij naar status {status}.",
            "Zet order {order_number} op {status}.",
            "Wijzig de status van {order_number} naar {status}.",
            "Pas bestelling {order_number} aan: nieuwe status {status}.",
            "Markeer order {order_number} als {status}.",
            "Registreer voor {order_number} de status {status}.",
        ]
    elif kind == 9:
        target, values = "create_event", {
            "event": _augmented_value("events", index), "date": _augmented_date(index),
            "venue": _augmented_value("venues", index),
        }
        templates = [
            "Maak een agenda-evenement '{event}' op {date} in {venue}.",
            "Plan '{event}' voor {date} bij {venue}.",
            "Zet {event} op {date} in mijn agenda; locatie {venue}.",
            "Voeg '{event}' toe aan {date}, plaats {venue}.",
            "Noteer {event} op {date} in {venue}.",
            "Maak een kalenderafspraak voor {event}, {date}, {venue}.",
        ]
    elif kind == 10:
        target, values = "move_robot", {
            "destination": _augmented_value("robot_destinations", index),
            "speed": _augmented_value("robot_speeds", index),
        }
        templates = [
            "Laat de robot naar {destination} rijden met snelheid {speed}.",
            "Stuur de robot naar {destination}; doe dat {speed}.",
            "Navigeer naar {destination} en beweeg {speed}.",
            "Robot, ga naar {destination} op tempo {speed}.",
        ]
    elif kind == 11:
        target, values = "pick_up_object", {
            "object": _augmented_value("robot_objects", index),
            "location": _augmented_value("robot_destinations", index + 3),
        }
        templates = [
            "Pak {object} op bij {location}.",
            "Neem {object} mee vanaf {location}.",
            "Kun je {object} oppakken in {location}?",
            "Haal {object} op; het ligt bij {location}.",
        ]
    elif kind == 12:
        target, values = "inspect_object", {"object": _augmented_value("robot_objects", index)}
        templates = [
            "Inspecteer {object}.", "Bekijk {object} nauwkeurig.",
            "Controleer of er iets mis is met {object}.",
            "Voer een inspectie uit van {object}.",
        ]
    elif kind == 13:
        text_value, _sentiment = _AUGMENTED_SENTIMENTS[index % len(_AUGMENTED_SENTIMENTS)]
        target, values = "analyze_sentiment", {"text": text_value}
        templates = [
            "Analyseer het sentiment van deze tekst: '{text}'.",
            "Is de volgende zin positief, negatief of neutraal? '{text}'",
            "Classificeer de toon van het bericht '{text}'.",
            "Bepaal het sentiment; de tekst luidt: {text}",
        ]
    elif kind == 14:
        target, values = "launch_game", {
            "game": _augmented_value("games", index),
            "platform": _augmented_value("platforms", index),
        }
        templates = [
            "Start {game} op {platform}.",
            "Open het spel {game} op mijn {platform}.",
            "Kun je {game} lanceren via {platform}?",
            "Ik wil {game} spelen op {platform}; start het.",
        ]
    elif kind == 15:
        target, values = "invite_player", {
            "player": _augmented_value("players", index),
            "game": _augmented_value("games", index + 2),
        }
        templates = [
            "Nodig speler {player} uit voor {game}.",
            "Stuur {player} een uitnodiging voor {game}.",
            "Laat {player} meedoen aan {game}.",
            "Voeg {player} toe aan mijn {game}-sessie.",
        ]
    elif kind == 16:
        target, values = "set_game_difficulty", {
            "game": _augmented_value("games", index),
            "difficulty": _augmented_value("difficulties", index),
        }
        templates = [
            "Zet {game} op moeilijkheid {difficulty}.",
            "Stel de moeilijkheidsgraad van {game} in op {difficulty}.",
            "Speel {game} op niveau {difficulty}.",
            "Maak {game} {difficulty}.",
        ]
    elif kind == 17:
        target, values = "report_player", {
            "player": _augmented_value("players", index),
            "reason": _augmented_value("report_reasons", index),
        }
        templates = [
            "Meld speler {player} wegens {reason}.",
            "Rapporteer {player}; reden: {reason}.",
            "Dien een klacht in over {player} voor {reason}.",
            "Markeer {player} vanwege {reason}.",
        ]
    elif kind == 18:
        target, values = "control_lights", {
            "room": _augmented_value("robot_rooms", index),
            "state": _augmented_value("robot_states", index),
        }
        templates = [
            "Doe de lichten in de {room} {state}.",
            "Zet de verlichting van de {room} op {state}.",
            "Maak de lampen in de {room} {state}.",
            "Regel het licht in de {room}: {state}.",
        ]
    elif kind == 19:
        target, values = "set_temperature", {
            "room": _augmented_value("robot_rooms", index),
            "temperature": _augmented_value("temperatures", index),
        }
        templates = [
            "Stel de temperatuur in de {room} in op {temperature}.",
            "Maak het in de {room} {temperature}.",
            "Zet de thermostaat van de {room} op {temperature}.",
            "Regel {room} naar {temperature}.",
        ]
    elif kind == 20:
        target, values = "book_ticket", {
            "destination": _augmented_value("cities", index + 5),
            "date": _augmented_date(index),
            "passenger": _augmented_value("passengers", index),
        }
        templates = [
            "Boek voor {passenger} een ticket naar {destination} op {date}.",
            "Reserveer een reis naar {destination} voor {passenger}, datum {date}.",
            "Plan {date} een ticket naar {destination} op naam van {passenger}.",
            "Koop een kaartje voor {passenger} naar {destination} op {date}.",
        ]
    elif kind == 21:
        target, values = "play_music", {
            "artist": _augmented_value("artists", index),
            "genre": _augmented_value("genres", index),
        }
        templates = [
            "Speel muziek van {artist} in het genre {genre}.",
            "Start {genre} van {artist}.",
            "Zet een nummer van {artist} op; genre {genre}.",
            "Ik wil {genre} luisteren, bijvoorbeeld {artist}.",
        ]
    elif kind == 22:
        target, values = "translate_text", {
            "text": _AUGMENTED_SENTIMENTS[index % len(_AUGMENTED_SENTIMENTS)][0],
            "target_language": _augmented_value("languages", index),
        }
        templates = [
            "Vertaal deze tekst naar {target_language}: '{text}'.",
            "Zet '{text}' om in {target_language}.",
            "Kun je dit vertalen? Doeltaal {target_language}: {text}",
            "Maak een {target_language} vertaling van: {text}",
        ]
    elif kind == 23:
        target, values = "set_robot_mode", {"mode": _augmented_value("robot_modes", index)}
        templates = [
            "Zet de robot in {mode}.",
            "Schakel de robot over naar de modus {mode}.",
            "Gebruik vanaf nu robotmodus {mode}.",
            "Activeer de {mode}-stand.",
        ]
    elif kind == 24:
        target, values = "send_email", {
            "recipient": _augmented_value("recipients", index),
            "subject": _augmented_value("email_subjects", index),
            "body": _augmented_value("messages", index),
        }
        templates = [
            "Stuur een e-mail naar {recipient} met onderwerp '{subject}' en tekst: {body}.",
            "Mail {recipient}; onderwerp: {subject}, inhoud: {body}.",
            "Kun je {recipient} mailen over {subject} dat {body}?",
            "Verstuur een e-mail aan {recipient} getiteld '{subject}' met de boodschap: {body}.",
        ]
    elif kind == 25:
        target, values = "create_note", {
            "title": _augmented_value("note_titles", index),
            "content": _augmented_value("messages", index),
        }
        templates = [
            "Maak een notitie getiteld '{title}' met inhoud: {content}.",
            "Noteer onder '{title}' het volgende: {content}.",
            "Sla een nieuwe notitie op met titel '{title}' en tekst '{content}'.",
            "Schrijf '{content}' op in een notitie genaamd '{title}'.",
        ]
    elif kind == 26:
        target, values = "add_todo_item", {
            "task": _augmented_value("tasks", index),
            "priority": _augmented_value("priorities", index),
        }
        templates = [
            "Voeg de taak '{task}' toe met prioriteit {priority}.",
            "Zet '{task}' op mijn to-do lijst; prioriteit is {priority}.",
            "Maak een actiepunt aan voor {task} en markeer het als {priority}.",
            "Plaats '{task}' op mijn takenlijst met status {priority}.",
        ]
    elif kind == 27:
        target, values = "delete_alarm", {"time": _augmented_time(index)}
        templates = [
            "Verwijder het alarm van {time}.",
            "Zet het alarm om {time} uit en wis het.",
            "Schakel de wekker van {time} uit.",
            "Annuleer het weksignaal voor {time}.",
        ]
    elif kind == 28:
        target, values = "cancel_appointment", {
            "date": _augmented_date(index),
            "contact": _augmented_value("names", index),
        }
        templates = [
            "Zeg de afspraak met {contact} op {date} af.",
            "Annuleer mijn ontmoeting op {date} met {contact}.",
            "Verwijder de afspraak van {date} met {contact} uit mijn agenda.",
            "Laat de afspraak op {date} met {contact} niet doorgaan.",
        ]
    elif kind == 29:
        target, values = "reschedule_meeting", {
            "date": _augmented_date(index),
            "new_time": _augmented_time(index),
            "person": _augmented_value("names", index),
        }
        templates = [
            "Verplaats de vergadering op {date} met {person} naar {new_time}.",
            "Wijzig de tijd van het overleg op {date} met {person} naar {new_time}.",
            "Plan de afspraak met {person} op {date} opnieuw in om {new_time}.",
            "Verschuif het gesprek van {date} met {person} naar tijdstip {new_time}.",
        ]
    elif kind == 30:
        target, values = "transfer_money", {
            "recipient": _augmented_value("names", index),
            "amount": _augmented_value("amounts", index),
            "description": _augmented_value("descriptions", index),
        }
        templates = [
            "Maak {amount} over naar {recipient} voor {description}.",
            "Kun je {amount} overmaken aan {recipient} met omschrijving '{description}'?",
            "Graag een overboeking van {amount} naar {recipient} wegens {description}.",
            "Wilt u {amount} overschrijven op naam van {recipient}, omschrijving: {description}?",
        ]
    elif kind == 31:
        target, values = "check_balance", {"account": _augmented_value("accounts", index)}
        templates = [
            "Wat is het saldo van mijn {account}?",
            "Toon het actuele saldo op de {account}.",
            "Hoeveel geld staat er momenteel op mijn {account}?",
            "Controleer het saldo van de {account}.",
        ]
    elif kind == 32:
        target, values = "pay_invoice", {
            "invoice_number": f"INV-2026-{index % 90000 + 10000:05d}",
            "amount": _augmented_value("amounts", index),
            "vendor": _augmented_value("names", index),
        }
        templates = [
            "Betaal factuur {invoice_number} van {amount} aan {vendor}.",
            "Voldoe factuurnummer {invoice_number} ten bedrage van {amount} voor {vendor}.",
            "Maak {amount} over voor factuur {invoice_number} gericht aan {vendor}.",
            "Verwerk de betaling van {invoice_number} ({amount}) aan {vendor}.",
        ]
    elif kind == 33:
        target, values = "lock_door", {
            "door": _augmented_value("doors", index),
            "status": "vergrendeld" if index % 2 == 0 else "ontgrendeld",
        }
        templates = [
            "Zet de {door} op {status}.",
            "Zorg ervoor dat de {door} wordt {status}.",
            "Maak de {door} {status}.",
            "Controleer en maak de {door} direct {status}.",
        ]
    elif kind == 34:
        target, values = "set_fan_speed", {
            "room": _augmented_value("robot_rooms", index),
            "fan_speed": _augmented_value("fan_speeds", index),
        }
        templates = [
            "Zet de ventilator in de {room} op {fan_speed}.",
            "Regel de ventilatie van de {room} naar {fan_speed}.",
            "Schakel de ventilator in {room} naar stand {fan_speed}.",
            "Verander de ventilatorsnelheid in de {room} naar {fan_speed}.",
        ]
    elif kind == 35:
        target, values = "open_curtains", {
            "room": _augmented_value("robot_rooms", index),
            "percentage": _augmented_value("curtain_percentages", index),
        }
        templates = [
            "Open de gordijnen in de {room} voor {percentage}.",
            "Zet de gordijnen van de {room} op {percentage}.",
            "Schuif de gordijnen in de {room} open tot {percentage}.",
            "Regel de zonwering in de {room} op {percentage}.",
        ]
    elif kind == 36:
        target, values = "book_hotel", {
            "hotel": _augmented_value("hotels", index),
            "city": _augmented_value("cities", index),
            "date": _augmented_date(index),
        }
        templates = [
            "Boek een kamer in {hotel} te {city} voor {date}.",
            "Reserveer bij {hotel} in {city} op datum {date}.",
            "Maak een hotelreservering in {hotel} ({city}) voor {date}.",
            "Plan een verblijf bij {hotel} in {city} op {date}.",
        ]
    elif kind == 37:
        target, values = "check_flight_status", {
            "flight_number": _augmented_value("flight_numbers", index),
            "airline": _augmented_value("airlines", index),
        }
        templates = [
            "Wat is de status van vlucht {flight_number} bij {airline}?",
            "Controleer vlucht {flight_number} van {airline}.",
            "Geef actuele reisinformatie over {airline} vlucht {flight_number}.",
            "Heeft vlucht {flight_number} ({airline}) vertraging?",
        ]
    elif kind == 38:
        target, values = "add_to_cart", {
            "product": _augmented_value("products", index),
            "quantity": index % 4 + 1,
        }
        templates = [
            "Voeg {quantity} stuks van {product} toe aan mijn winkelwagen.",
            "Plaats {quantity} keer {product} in het winkelmandje.",
            "Ik wil {quantity} stuks van {product} in mijn mandje leggen.",
            "Bestel {quantity} stuks van {product} via mijn winkelwagen.",
        ]
    elif kind == 39:
        target, values = "cancel_order", {
            "order_number": f"ORD-{index % 900000 + 100000:07d}",
            "reason": _augmented_value("reasons", index),
        }
        templates = [
            "Annuleer order {order_number} wegens {reason}.",
            "Zet bestelling {order_number} stop om reden {reason}.",
            "Ik wil bestelling {order_number} annuleren vanwege {reason}.",
            "Verwijder order {order_number}; reden is {reason}.",
        ]
    elif kind == 40:
        target, values = "request_refund", {
            "order_number": f"ORD-{index % 900000 + 100000:07d}",
            "amount": _augmented_value("amounts", index),
        }
        templates = [
            "Vraag een terugbetaling van {amount} aan voor order {order_number}.",
            "Dien een restitutieverzoek in van {amount} voor bestelling {order_number}.",
            "Ik wil {amount} retour ontvangen voor order {order_number}.",
            "Vorder {amount} terug van bestelling {order_number}.",
        ]
    elif kind == 41:
        target, values = "check_stock", {
            "product": _augmented_value("products", index),
            "store": _augmented_value("stores", index),
        }
        templates = [
            "Is {product} op voorraad bij {store}?",
            "Controleer de voorraad van {product} in {store}.",
            "Kijk na of {product} beschikbaar is bij {store}.",
            "Toon de beschikbaarheid van {product} in vestiging {store}.",
        ]
    elif kind == 42:
        target, values = "pause_playback", {"device": _augmented_value("devices", index)}
        templates = [
            "Pauzeer het afspelen op de {device}.",
            "Zet de weergave op de {device} op pauze.",
            "Stop tijdelijk de muziek op {device}.",
            "Pauzeer audio op mijn {device}.",
        ]
    elif kind == 43:
        target, values = "adjust_volume", {
            "device": _augmented_value("devices", index),
            "level": _augmented_value("volume_levels", index),
        }
        templates = [
            "Zet het volume van de {device} op {level}.",
            "Pas het geluidsniveau van {device} aan naar {level}.",
            "Regel het volume op {device} naar {level}.",
            "Stel het volume van de {device} in op {level}.",
        ]
    elif kind == 44:
        target, values = "create_playlist", {
            "playlist_name": _augmented_value("playlist_names", index),
            "genre": _augmented_value("genres", index),
        }
        templates = [
            "Maak een nieuwe afspeellijst '{playlist_name}' voor genre {genre}.",
            "Creëer playlist '{playlist_name}' met muziekstijl {genre}.",
            "Stel een lijst '{playlist_name}' samen binnen het genre {genre}.",
            "Start een nieuwe muzieklijst getiteld '{playlist_name}' ({genre}).",
        ]
    elif kind == 45:
        target, values = "schedule_doctor_visit", {
            "doctor": _augmented_value("doctors", index),
            "date": _augmented_date(index),
            "time": _augmented_time(index),
        }
        templates = [
            "Maak een consultatieafspraak bij {doctor} op {date} om {time}.",
            "Plan een doktersbezoek bij {doctor} voor {date} om {time}.",
            "Boek een consult bij {doctor} op {date}, tijdstip {time}.",
            "Reserveer een medische afspraak met {doctor} voor {date} om {time}.",
        ]
    elif kind == 46:
        target, values = "log_symptoms", {
            "symptom": _augmented_value("symptoms", index),
            "severity": _augmented_value("severities", index),
        }
        templates = [
            "Registreer symptoom {symptom} met ernst {severity}.",
            "Noteer in mijn gezondheidslogboek {symptom} als {severity}.",
            "Sla de klacht {symptom} op; niveau is {severity}.",
            "Houd bij dat ik last heb van {symptom} ({severity}).",
        ]
    elif kind == 47:
        target, values = "find_route", {
            "origin": _augmented_value("cities", index),
            "destination": _augmented_value("cities", index + 6),
        }
        templates = [
            "Zoek een routebeschrijving van {origin} naar {destination}.",
            "Hoe reis ik het snelst van {origin} naar {destination}?",
            "Toon de routeplanner van {origin} naar {destination}.",
            "Navigeer vanaf {origin} in de richting van {destination}.",
        ]
    elif kind == 48:
        target, values = "summarize_document", {"title": _augmented_value("titles", index)}
        templates = [
            "Vat het document '{title}' kort samen.",
            "Geef een samenvatting van de tekst '{title}'.",
            "Maak een beknopt overzicht van het stuk '{title}'.",
            "Kun je '{title}' samenvatten in enkele kernpunten?",
        ]
    elif kind == 49:
        target, values = "toggle_device", {
            "device_name": _augmented_value("devices", index),
            "state": "aan" if index % 2 == 0 else "uit",
        }
        templates = [
            "Zet de {device_name} {state}.",
            "Schakel het apparaat {device_name} {state}.",
            "Maak de {device_name} {state}.",
            "Doe de {device_name} {state}.",
        ]
    elif kind == 50:
        target, values = "archive_email", {
            "subject": _augmented_value("email_subjects", index),
            "sender": _augmented_value("recipients", index),
        }
        templates = [
            "Archiveer het bericht met onderwerp '{subject}' van afzender {sender}.",
            "Plaats de e-mail '{subject}' van {sender} in het archief.",
            "Verplaats het bericht van {sender} over {subject} naar het archief.",
            "Sla de e-mail '{subject}' ({sender}) op in het archief.",
        ]
    else:
        target, values = "set_reminder", {
            "reminder": _augmented_value("reminders", index),
            "time": _augmented_time(index),
        }
        templates = [
            "Stel een herinnering in voor {reminder} om {time}.",
            "Herinner me er om {time} aan: {reminder}.",
            "Maak een herinnering met tekst '{reminder}' voor {time}.",
            "Zorg dat ik om {time} een herinnering krijg om {reminder}.",
        ]

    query = _augmented_context(
        templates[index % len(templates)].format(**values), index
    ) + f" Referentie {index}."
    if locale == "nl-BE":
        query = "Kunt ge dit regelen? " + query
    elif locale == "nl-noisy":
        # Apply noise only through the value-aware typo helper.  Blind string
        # replacements can alter an argument embedded in a message/title and
        # make an otherwise correct label fail grounding validation.
        query = "pls " + _augmented_typo(query.casefold(), values)
    return {
        "id": f"augmented:{split}:{index}", "group_id": f"augmented:{index}",
        "source": "deterministic-dutch-augmentation-v1", "locale": locale,
        "domain": target.split("_")[-1], "schema_variant": variant,
        "tool_similarity": "mixed_catalogue", "task_family": "action", "split": split,
        "query": query, "tools": _augmented_catalogue(target, index, variant),
        "answers": [{"name": target, "arguments": values}],
        "reasoning": "; ".join(f"{field}='{value}' uit de vraag" for field, value in values.items()),
    }


_AUGMENTED_FIELD_LABELS = {
    "name": "naam", "email": "e-mailadres", "phone": "telefoonnummer",
    "date": "datum", "time": "tijd", "location": "locatie",
    "vendor": "leverancier", "invoice_number": "factuurnummer", "total": "totaal",
    "order_number": "ordernummer", "product": "product", "quantity": "aantal",
    "tracking_number": "trackingnummer", "carrier": "vervoerder", "status": "status",
    "price": "prijs", "availability": "beschikbaarheid", "author": "auteur",
    "title": "titel", "year": "jaar", "address": "adres", "postal_code": "postcode",
    "sender": "afzender", "subject": "onderwerp", "message": "bericht",
    "event": "evenement", "venue": "locatie", "merchant": "winkel",
    "currency": "valuta", "category": "categorie", "recipient": "ontvanger",
    "item": "artikel", "unit_price": "stukprijs", "doctor": "arts",
    "specialty": "specialisme", "iban": "rekeningnummer", "amount": "bedrag",
    "description": "omschrijving", "hotel": "hotel", "city": "stad",
    "checkin": "incheckdatum", "nights": "nachten", "airline": "luchtvaartmaatschappij",
    "flight_number": "vluchtnummer", "origin": "vertreklocatie", "destination": "bestemming",
    "customer": "klant", "issue": "probleem", "priority": "prioriteit",
    "order_id": "bestelnummer", "reason": "reden",
}


def _augmented_extraction(index: int, split: str) -> dict[str, Any]:
    name, fields = _EXTRACTION_SCHEMAS[(index // 10) % len(_EXTRACTION_SCHEMAS)]
    values = _extraction_values(name, fields, index + 3_000_000)
    locale = _augmented_locale(index)
    variant = 2 if split == "test" else index % 2
    labels = {_AUGMENTED_FIELD_LABELS.get(field, field): value for field, value in values.items()}
    body = ". ".join(f"{label.capitalize()}: {value}" for label, value in labels.items()) + "."
    prompts = [
        "Haal de gevraagde gegevens uit deze tekst: ",
        "Zet de informatie hieronder om naar velden: ",
        "Lees dit bericht en extraheer de relevante waarden: ",
        "Welke gegevens staan er in de volgende tekst? ",
        "Verwerk dit formulier nauwkeurig: ",
        "Maak hier een gestructureerde registratie van: ",
        "Haal alle relevante gegevens op uit het overzicht: ",
        "Zet de gegevens uit de onderstaande passage om naar een registratie: ",
    ]
    query = _augmented_context(prompts[index % len(prompts)] + body, index) + f" Referentie {index}."
    if locale == "nl-BE":
        query = "Kunt ge dit even verwerken? " + query
    elif locale == "nl-noisy":
        query = "pls " + _augmented_typo(query.casefold(), values)
    tools = [_extraction_tool(name, fields, variant)]
    distractors = [
        (tool_name, tool_fields) for tool_name, tool_fields in _EXTRACTION_SCHEMAS
        if tool_name != name
    ]
    random.Random(index * 15485863).shuffle(distractors)
    tools.extend(_extraction_tool(tool_name, tool_fields, variant)
                 for tool_name, tool_fields in distractors[:3])
    random.Random(index * 15485863 + 1).shuffle(tools)
    return {
        "id": f"augmented-extraction:{split}:{index}",
        "group_id": f"augmented-extraction:{index}",
        "source": "deterministic-dutch-augmentation-v1", "locale": locale,
        "domain": "extraction", "extraction_type": name, "schema_variant": variant,
        "tool_similarity": "similar_record_tools", "task_family": "extraction", "split": split,
        "query": query, "tools": tools,
        "answers": [{"name": name, "arguments": values}],
        "reasoning": "; ".join(f"{field}='{value}' uit de tekst" for field, value in values.items()),
    }


def _augmented_negative(index: int, split: str) -> dict[str, Any]:
    topics = [
        "Leg uit waarom de maan soms overdag zichtbaar is.",
        "Schrijf een korte samenvatting van een fictieve roman.",
        "Wat betekent het Nederlandse woord 'zorgvuldig'?",
        "Vergelijk fietsen in Amsterdam en Brussel.",
        "Ik denk hardop na over een alarm om {time}; stel niets in.",
        "Mijn pakket {tracking} is al bezorgd, doe er niets mee.",
        "Kun je uitleggen hoe {minutes} minuten in een uur passen?",
        "Welke kleur past bij een rustige werkkamer.",
        "Vertel me meer over de geschiedenis van de Noordzee (vraag {index}).",
        "Wat zijn bekende bezienswaardigheden in de provincie Utrecht?",
        "Ik wilde een wekker zetten voor {time}, maar laat maar zitten.",
        "De bestelling {tracking} is al geannuleerd, geen actie nodig.",
        "Geef een grammaticale analyse van een samengestelde zin (versie {index}).",
        "Hoeveel liter water drinkt een gemiddelde volwassene per dag?",
        "Activeer géén herinnering om {time}, ik noteer het zelf wel.",
        "Wat is het verschil tussen een desktop en een laptop bij vraag {index}?",
    ]
    values = {
        "time": _augmented_time(index), "tracking": f"3SNL{index % 90000000000 + 10000000000:011d}",
        "minutes": 5 + index % 176,
    }
    query = _augmented_context(
        topics[index % len(topics)].format(**values, index=index) + f" (verzoek {index})", index)
    locale = _augmented_locale(index)
    if locale == "nl-BE":
        query = "Kunt ge dit toelichten? " + query
    elif locale == "nl-noisy":
        query = "pls " + _augmented_typo(query.casefold(), values)
    return {
        "id": f"augmented-negative:{split}:{index}", "group_id": f"augmented-negative:{index}",
        "source": "deterministic-dutch-augmentation-v1", "locale": locale,
        "domain": "general", "schema_variant": 2 if split == "test" else index % 2,
        "tool_similarity": "hard_negative", "task_family": "negative", "split": split,
        "query": query,
        "tools": [_augmented_tool(name, _AUGMENTED_TOOL_SPECS[name], index % 3)
                  for name in ("set_alarm", "get_weather", "send_message", "track_shipment", "control_lights")[:3]],
        "answers": [],
    }


def _augmented_multi(index: int, split: str) -> dict[str, Any]:
    scenario = (index // 7) % 6
    locale = _augmented_locale(index)
    variant = 2 if split == "test" else index % 2

    if scenario == 0:
        time = _augmented_time(index)
        minutes = 5 + index % 176
        values = {"time": time, "minutes": minutes}
        queries = [
            "Zet een alarm om {time} en start een timer van {minutes} minuten (verzoek {index}).",
            "Plan een timer van {minutes} minuten en maak me om {time} wakker (verzoek {index}).",
            "Ik wil beide: een alarm voor {time} en een timer van {minutes} minuten (verzoek {index}).",
            "Stel een alarm in om {time}; gebruik ook een timer van {minutes} minuten (verzoek {index}).",
        ]
        tools = ["set_alarm", "set_timer", "get_weather"]
        answers = [
            {"name": "set_alarm", "arguments": {"time": time}},
            {"name": "set_timer", "arguments": {"minutes": minutes}},
        ]
        reasoning = f"time='{time}' en minutes='{minutes}' uit de vraag"
        domain = "time"
    elif scenario == 1:
        room = _augmented_value("robot_rooms", index)
        state = _augmented_value("robot_states", index)
        temp = _augmented_value("temperatures", index)
        values = {"room": room, "state": state, "temperature": temp}
        queries = [
            "Doe de lichten in de {room} {state} en zet de temperatuur op {temperature} (verzoek {index}).",
            "Schakel de lampen in {room} {state} en stel de thermostaat in op {temperature} (verzoek {index}).",
            "Regel het klimaat en licht: {room} lampen {state}, temperatuur naar {temperature} (verzoek {index}).",
        ]
        tools = ["control_lights", "set_temperature", "lock_door"]
        answers = [
            {"name": "control_lights", "arguments": {"room": room, "state": state}},
            {"name": "set_temperature", "arguments": {"room": room, "temperature": temp}},
        ]
        reasoning = f"room='{room}', state='{state}' en temperature='{temp}' uit de vraag"
        domain = "smart_home"
    elif scenario == 2:
        date_val = _augmented_date(index)
        time_val = _augmented_time(index)
        loc_val = _augmented_value("locations", index)
        recipient = _augmented_value("recipients", index)
        msg = _augmented_value("messages", index)
        values = {"date": date_val, "time": time_val, "location": loc_val, "recipient": recipient, "message": msg}
        queries = [
            "Plan een afspraak op {date} om {time} op {location} en stuur naar {recipient}: {message} (verzoek {index}).",
            "Zet de ontmoeting vast op {date} om {time} bij {location} en laat {recipient} weten dat {message} (verzoek {index}).",
        ]
        tools = ["create_appointment", "send_message", "find_contact"]
        answers = [
            {"name": "create_appointment", "arguments": {"date": date_val, "time": time_val, "location": loc_val}},
            {"name": "send_message", "arguments": {"recipient": recipient, "message": msg}},
        ]
        reasoning = f"date='{date_val}', time='{time_val}', location='{loc_val}', recipient='{recipient}' uit de vraag"
        domain = "coordination"
    elif scenario == 3:
        title = _augmented_value("note_titles", index)
        content = _augmented_value("messages", index)
        task = _augmented_value("tasks", index)
        priority = _augmented_value("priorities", index)
        values = {"title": title, "content": content, "task": task, "priority": priority}
        queries = [
            "Maak een notitie getiteld '{title}' met inhoud '{content}' en voeg de taak '{task}' toe met prioriteit {priority} (verzoek {index}).",
            "Noteer '{content}' onder titel '{title}' en zet tevens '{task}' op mijn takenlijst met status {priority} (verzoek {index}).",
        ]
        tools = ["create_note", "add_todo_item", "send_email"]
        answers = [
            {"name": "create_note", "arguments": {"title": title, "content": content}},
            {"name": "add_todo_item", "arguments": {"task": task, "priority": priority}},
        ]
        reasoning = f"title='{title}', content='{content}', task='{task}' en priority='{priority}' uit de vraag"
        domain = "productivity"
    elif scenario == 4:
        door = _augmented_value("doors", index)
        door_status = "vergrendeld" if index % 2 == 0 else "ontgrendeld"
        room = _augmented_value("robot_rooms", index)
        light_state = "uit" if index % 2 == 0 else "aan"
        values = {"door": door, "door_status": door_status, "room": room, "light_state": light_state}
        queries = [
            "Zet de {door} op {door_status} en doe de lichten in de {room} {light_state} (verzoek {index}).",
            "Zorg dat de {door} wordt {door_status} en schakel de verlichting in de {room} {light_state} (verzoek {index}).",
        ]
        tools = ["lock_door", "control_lights", "set_fan_speed"]
        answers = [
            {"name": "lock_door", "arguments": {"door": door, "status": door_status}},
            {"name": "control_lights", "arguments": {"room": room, "state": light_state}},
        ]
        reasoning = f"door='{door}', status='{door_status}', room='{room}' en state='{light_state}' uit de vraag"
        domain = "security"
    else:
        artist = _augmented_value("artists", index)
        genre = _augmented_value("genres", index)
        device = _augmented_value("devices", index)
        level = _augmented_value("volume_levels", index)
        values = {"artist": artist, "genre": genre, "device": device, "level": level}
        queries = [
            "Speel muziek van {artist} in het genre {genre} en zet het volume van de {device} op {level} (verzoek {index}).",
            "Start {genre} van {artist} en regel het volume op {device} naar {level} (verzoek {index}).",
        ]
        tools = ["play_music", "adjust_volume", "pause_playback"]
        answers = [
            {"name": "play_music", "arguments": {"artist": artist, "genre": genre}},
            {"name": "adjust_volume", "arguments": {"device": device, "level": level}},
        ]
        reasoning = f"artist='{artist}', genre='{genre}', device='{device}' en level='{level}' uit de vraag"
        domain = "media"

    rng = random.Random(index * 104729)
    rng.shuffle(tools)
    query = _augmented_context(
        queries[index % len(queries)].format(**values, index=index), index)
    if locale == "nl-BE":
        query = "Kunt ge dit regelen? " + query
    elif locale == "nl-noisy":
        query = "pls " + _augmented_typo(query.casefold(), values)
    return {
        "id": f"augmented-multi:{split}:{index}", "group_id": f"augmented-multi:{index}",
        "source": "deterministic-dutch-augmentation-v1", "locale": locale,
        "domain": domain, "schema_variant": variant, "tool_similarity": "competing_workflow_tools",
        "task_family": "multi_call", "split": split,
        "query": query,
        "tools": [_augmented_tool(name, _AUGMENTED_TOOL_SPECS[name], variant) for name in tools],
        "answers": answers,
        "reasoning": reasoning,
    }


def make_augmented_examples(count: int, *, split: str = "train", seed: int = 0,
                            start: int = 0) -> list[dict[str, Any]]:
    """Generate many varied, deterministic Dutch records without a hosted model.

    The mix is 60% single-call actions, 20% extraction, 10% no-call hard
    negatives, and 10% multi-call requests.  ``seed`` changes both vocabulary
    values and templates while preserving valid labels and grounding.
    """
    rows = []
    for offset in range(count):
        index = start + offset + seed * 1_000_000
        bucket = index % 10
        if bucket < 6:
            rows.append(_augmented_action(index, split))
        elif bucket < 8:
            rows.append(_augmented_extraction(index, split))
        elif bucket == 8:
            rows.append(_augmented_negative(index, split))
        else:
            rows.append(_augmented_multi(index, split))
    return rows


def augment_training_file(input_path: str | Path, output_path: str | Path, *, count: int,
                          seed: int = 0) -> dict[str, Any]:
    """Copy an existing training JSONL and append a reproducible augmentation set."""
    base = read_jsonl(input_path)
    extra = make_augmented_examples(count, split="train", seed=seed, start=len(base) + 10_000)
    combined = base + extra
    random.Random(seed).shuffle(combined)
    output = write_jsonl(output_path, combined)
    manifest = {
        "source": str(input_path), "source_sha256": _file_sha256(input_path),
        "output": output, "seed": seed, "added_examples": len(extra),
        "total_examples": len(combined), "generator": "deterministic-dutch-augmentation-v1",
    }
    manifest_path = Path(output_path).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def create_dutch_dataset(massive_nl_path: str | Path, output_dir: str | Path, *,
                         massive_en_path: str | Path | None = None, seed: int = 0,
                         action_count: int = 8000, extraction_count: int = 2000,
                         negative_count: int = 1200, multi_count: int = 600,
                         english_count: int = 1200,
                         augmentation_count: int = 0) -> dict[str, str]:
    """Build reproducible Dutch train/validation/test JSONL files from MASSIVE.

    ``massive_en_path`` is required when English replay is requested.  The
    function never downloads data, keeping both licences and network use explicit.
    """
    nl_rows = read_jsonl(massive_nl_path)
    en_rows = read_jsonl(massive_en_path) if massive_en_path else []
    if english_count and not en_rows:
        raise ValueError("pass massive_en_path to create English replay examples")
    output_dir = Path(output_dir)
    train = build_massive_examples(nl_rows, partition="train", limit=action_count,
                                   seed=seed, split="train")
    train += make_extraction_examples(extraction_count, split="train", seed=seed)
    train += make_negative_examples(negative_count, split="train", seed=seed)
    train += make_multi_call_examples(multi_count, split="train", seed=seed)
    if english_count:
        train += build_massive_examples(en_rows, partition="train", limit=english_count,
                                        seed=seed, split="train")
    if augmentation_count:
        train += make_augmented_examples(
            augmentation_count, split="train", seed=seed, start=len(train) + 10_000
        )
    rng = random.Random(seed)
    rng.shuffle(train)

    val = build_massive_examples(nl_rows, partition="dev", seed=seed, split="val")
    val += make_extraction_examples(100, split="val", seed=seed, start=1_000_000)
    val += make_negative_examples(60, split="val", seed=seed, start=1_000_000)
    val += make_multi_call_examples(40, split="val", seed=seed, start=1_000_000)
    test = build_massive_examples(nl_rows, partition="test", seed=seed, split="test")
    test += make_extraction_examples(160, split="test", seed=seed, start=2_000_000)
    test += make_negative_examples(120, split="test", seed=seed, start=2_000_000)
    test += make_multi_call_examples(120, split="test", seed=seed, start=2_000_000)
    english_val = build_massive_examples(en_rows, partition="dev", seed=seed, split="english_val") if en_rows else []
    review = [row | {"review_required": True} for row in val + test
              if row["source"].startswith("deterministic-dutch")]

    outputs = {
        "train": write_jsonl(output_dir / "train.jsonl", train),
        "val": write_jsonl(output_dir / "val.jsonl", val),
        "test": write_jsonl(output_dir / "test.jsonl", test),
        "english_val": write_jsonl(output_dir / "english_val.jsonl", english_val),
        "review": write_jsonl(output_dir / "review.jsonl", review),
    }
    manifest = {
        "seed": seed,
        "source": "MASSIVE-1.0 (CC BY 4.0) + deterministic-dutch-v1",
        "augmentation_count": augmentation_count,
        "augmentation_generator": (
            "deterministic-dutch-augmentation-v1" if augmentation_count else None
        ),
        "input_sha256": {"massive_nl": _file_sha256(massive_nl_path),
                         "massive_en": _file_sha256(massive_en_path)},
        "outputs": outputs, "counts": {name: len(read_jsonl(path)) for name, path in outputs.items()},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs["manifest"] = str(manifest_path)
    return outputs


def evaluate_runtime(path: str | Path, *, weights: str | None = None, limit: int = 0,
                     max_new_tokens: int = 256, bootstrap_samples: int = 1000,
                     seed: int = 0) -> dict[str, Any]:
    """Evaluate through Needle's deployed engine, not unconstrained JAX decoding."""
    from . import Needle

    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    agents: dict[str, Needle] = {}
    predictions = []
    invalid_calls = []
    for row in rows:
        tools = row.get("tools", [])
        tools_key = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        agent = agents.get(tools_key)
        if agent is None:
            agent = Needle(tools=tools, weights=weights)
            agents[tools_key] = agent
        response = agent.complete(str(row["query"]), max_new_tokens=max_new_tokens)
        calls = response.get("function_calls", [])
        if not isinstance(calls, list):
            invalid_calls.append({"id": row.get("id"), "errors": ["engine returned non-list function_calls"]})
            calls = []
        else:
            errors = validate_example({**row, "answers": calls}, require_grounding=True)
            if errors:
                invalid_calls.append({"id": row.get("id"), "errors": errors})
        predictions.append(calls)
    report = score_predictions(rows, predictions, bootstrap_samples=bootstrap_samples, seed=seed)
    report.update({"weights": weights, "dataset": str(path), "unique_tool_catalogues": len(agents),
                   "invalid_generated_calls": len(invalid_calls),
                   "all_generated_calls_schema_valid": not invalid_calls,
                   "invalid_call_examples": invalid_calls})
    return report


def release_gates(report: dict[str, Any], *, compact_report: dict[str, Any] | None = None,
                  english_baseline: dict[str, Any] | None = None,
                  english_tuned: dict[str, Any] | None = None,
                  second_seed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate the Dutch release criteria against engine-generated reports."""
    failures = []
    exact = report.get("exact_call_accuracy", 0.0)
    ci_low = (report.get("exact_call_ci95") or [0.0])[0]
    if exact < 0.85:
        failures.append(f"exact-call accuracy {exact:.3f} is below 0.850")
    if ci_low < 0.82:
        failures.append(f"exact-call CI lower bound {ci_low:.3f} is below 0.820")
    if report.get("tool_selection_accuracy", 0.0) < 0.90:
        failures.append("tool-selection accuracy is below 0.900")
    no_call = report.get("no_call") or {}
    if no_call.get("precision", 0.0) < 0.90 or no_call.get("recall", 0.0) < 0.90:
        failures.append("no-call precision or recall is below 0.900")
    if report.get("hallucinated_argument_rate", 1.0) >= 0.02:
        failures.append("hallucinated-argument rate is at least 0.020")
    if not report.get("all_generated_calls_schema_valid", False):
        failures.append("one or more generated calls fail schema or grounding validation")
    for dimension, values in (report.get("slices") or {}).items():
        for name, value in values.items():
            if value.get("examples", 0) and value.get("exact_call_accuracy", 0.0) < 0.75:
                failures.append(f"slice {dimension}={name} is below 0.750")
    if compact_report is not None:
        drop = exact - compact_report.get("exact_call_accuracy", 0.0)
        if drop > 0.010000001:
            failures.append(f"compact quantization drops exact-call accuracy by {drop:.3f}")
    if english_baseline is not None and english_tuned is not None:
        regression = english_baseline.get("exact_call_accuracy", 0.0) - english_tuned.get("exact_call_accuracy", 0.0)
        if regression > 0.05:
            failures.append(f"English exact-call regression is {regression:.3f}")
    if second_seed is not None:
        spread = abs(exact - second_seed.get("exact_call_accuracy", 0.0))
        if spread > 0.02:
            failures.append(f"seed exact-call spread is {spread:.3f}")
    return {"passed": not failures, "failures": failures,
            "exact_call_accuracy": exact, "ci_lower": ci_low}
