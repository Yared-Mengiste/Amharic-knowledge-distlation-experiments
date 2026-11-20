"""Explicit automated proxies and uncertainty; semantic quality needs human scores."""
import json
import ast
import csv
import io
import math
import re
from collections import Counter

from teacher.filter_teacher import check_math, geez_share, max_line_repeat, repetition_score

CONFIGURATIONS = ("llama_original", "llama_adapted", "gemma_original", "gemma_adapted")


def wilson(successes, total):
    if not total:
        return {"successes": 0, "total": 0, "rate": None, "ci95": None}
    z, p = 1.959963984540054, successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"successes": successes, "total": total, "rate": p, "ci95": [max(0, centre - half), min(1, centre + half)]}


def constraint_result(answer, rules):
    if not rules:
        return None
    checks = []
    for rule in rules:
        kind = rule["kind"]
        if kind in {"json", "json_keys", "json_type"}:
            try:
                parsed = json.loads(answer)
                if kind == "json":
                    passed = True
                elif kind == "json_keys":
                    passed = isinstance(parsed, dict) and set(parsed) == set(rule["keys"])
                else:
                    passed = isinstance(parsed, dict if rule["type"] == "object" else list)
            except json.JSONDecodeError:
                passed = False
        elif kind == "line_count":
            passed = len([line for line in answer.splitlines() if line.strip()]) == rule["count"]
        elif kind == "bullet_count":
            passed = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", answer)) == rule["count"]
        elif kind == "required_text":
            passed = all(text in answer for text in rule["texts"])
        elif kind == "forbidden_text":
            passed = not any(text in answer for text in rule["texts"])
        elif kind == "regex":
            passed = bool(re.fullmatch(rule["pattern"], answer, flags=re.DOTALL))
        elif kind == "csv_columns":
            try:
                rows = list(csv.reader(io.StringIO(answer), strict=True))
                passed = rule["min_rows"] <= len(rows) <= rule["max_rows"] and all(len(row) == rule["columns"] for row in rows)
            except csv.Error:
                passed = False
        elif kind == "markdown_table":
            lines = [line.strip() for line in answer.splitlines() if line.strip()]
            rows = [line.strip("|").split("|") for line in lines]
            passed = len(rows) == rule["data_rows"] + 2 and all(len(row) == rule["columns"] for row in rows)
            if passed:
                passed = all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in rows[1])
        elif kind == "yaml_fields":
            # A deliberately limited field/shape proxy; semantic YAML validity is manual.
            passed = all(len(re.findall(r"(?m)^\s*(?:-\s*)?" + re.escape(field) + r"\s*:\s*\S.*$", answer)) == rule["records"] for field in rule["fields"])
        elif kind == "python_syntax":
            source = answer.strip()
            if source.startswith("```python\n") and source.endswith("\n```"):
                source = source[len("```python\n"):-len("\n```")]
            try:
                parsed = ast.parse(source)
                passed = bool(parsed.body) and not (len(parsed.body) == 1 and isinstance(parsed.body[0], ast.Expr) and isinstance(parsed.body[0].value, ast.Constant) and isinstance(parsed.body[0].value.value, str))
            except SyntaxError:
                passed = False
        else:
            raise ValueError(f"Unsupported instruction constraint: {kind}")
        checks.append(bool(passed))
    return all(checks)


def automatic_metrics(output, math_checks, constraints):
    answer, category = output["answer"], output["category"]
    share = geez_share(answer)
    language = None
    if share is not None and category != "format":
        language = share <= 0.05 if category == "translation_am_en" else share >= (0.45 if category in {"code_switch", "latin_amharic", "english_instruction"} else 0.70)
    math_proxy = None
    if category == "math" and output["id"] in math_checks:
        result, _ = check_math(answer, math_checks[output["id"]])
        math_proxy = None if result == "manual" else result == "pass"
    return {"language_script_consistency": language,
            "math_reference_check_proxy": math_proxy,
            "format_instruction_constraints": constraint_result(answer, constraints.get(output["id"], [])),
            "truncated": output["termination_reason"] == "length",
            "repetition": repetition_score(answer) > 0.25 or max_line_repeat(answer) >= 3,
            "unexpected_stop": output["termination_reason"] == "unexpected_stop"}


def summarize_metrics(rows):
    result = {}
    for name in ("language_script_consistency", "math_reference_check_proxy", "format_instruction_constraints", "truncated", "repetition", "unexpected_stop"):
        values = [r[name] for r in rows if r[name] is not None]
        result[name] = wilson(sum(values), len(values))
        result[name]["unscored"] = len(rows) - len(values)
    return result
