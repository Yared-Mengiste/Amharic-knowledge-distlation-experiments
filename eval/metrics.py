"""Explicit automated proxies and uncertainty; semantic quality needs human scores."""
import json
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
        if kind in {"json", "json_keys"}:
            try:
                parsed = json.loads(answer)
                passed = True if kind == "json" else isinstance(parsed, dict) and set(parsed) == set(rule["keys"])
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
