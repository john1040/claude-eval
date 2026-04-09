"""Report: format and display evaluation results."""

import json
from pathlib import Path


def compute_summary(report_data: dict) -> dict:
    """Compute aggregate scores from case results."""
    criteria = report_data["criteria"]
    case_results = report_data["case_results"]

    if not case_results:
        return {"criteria_averages": {}, "weighted_score": 0.0, "weighted_pct": 0.0}

    criteria_averages = {}
    for c in criteria:
        name = c["name"]
        scores = [
            case["scores"][name]["score"]
            for case in case_results
            if name in case.get("scores", {}) and case["scores"][name].get("score", 0) > 0
        ]
        criteria_averages[name] = round(sum(scores) / len(scores), 1) if scores else 0.0

    total_weight = sum(c.get("weight", 1.0) for c in criteria)
    weighted_sum = sum(
        criteria_averages.get(c["name"], 0) * c.get("weight", 1.0)
        for c in criteria
    )
    weighted_score = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0

    return {
        "criteria_averages": criteria_averages,
        "weighted_score": weighted_score,
        "weighted_pct": round((weighted_score / 5.0) * 100, 1),
    }


def _usage_line(usage: dict) -> str:
    tokens_in = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)
    cost = usage.get("estimated_cost_usd", 0.0)
    return f"  Tokens: {tokens_in:,} in / {tokens_out:,} out  (~${cost:.4f})"


def print_report(report_data: dict):
    """Print a formatted CLI report."""
    eval_name = report_data["eval_name"]
    model = report_data["model"]
    num_cases = report_data["num_cases"]
    elapsed = report_data["elapsed_seconds"]
    criteria = report_data["criteria"]
    usage = report_data.get("usage", {})

    summary = compute_summary(report_data)

    line = "═" * 50
    print(f"\n  {line}")
    print(f"  claude-eval results: {eval_name}")
    print(f"  {line}\n")
    print(f"  Model:  {model}")
    print(f"  Cases:  {num_cases}")
    print(f"  Time:   {elapsed}s")
    if usage:
        print(_usage_line(usage))
    print()

    name_width = max(len(c["name"]) for c in criteria) + 2
    print(f"  {'Criteria':<{name_width}} {'Avg':>7}  {'Weight':>6}")
    print(f"  {'─' * name_width} {'─' * 7}  {'─' * 6}")

    for c in criteria:
        name = c["name"]
        avg = summary["criteria_averages"].get(name, 0.0)
        weight = c.get("weight", 1.0)
        print(f"  {name:<{name_width}} {avg:>5}/5  {weight:>6}")

    print()
    ws = summary["weighted_score"]
    wp = summary["weighted_pct"]
    print(f"  Weighted Score: {ws} / 5.0 ({wp}%)")
    print()

    for i, case_result in enumerate(report_data["case_results"]):
        scores_str = ", ".join(
            f"{k}={v.get('score', '?')}"
            for k, v in case_result.get("scores", {}).items()
        )
        print(f"  Case {i+1}: {scores_str}")

    print()


def print_comparison_report(comparison: dict):
    """Print a side-by-side multi-model comparison table.

    Args:
        comparison: dict with keys:
            eval_name, criteria, elapsed_seconds,
            models: list of model name strings,
            results: {model: {"case_results": [...], "usage": {...}}}
    """
    eval_name = comparison["eval_name"]
    criteria = comparison["criteria"]
    models = comparison["models"]
    elapsed = comparison["elapsed_seconds"]

    line = "═" * 60
    print(f"\n  {line}")
    print(f"  claude-eval comparison: {eval_name}")
    print(f"  {line}\n")
    print(f"  Cases:  {comparison['num_cases']}    Time: {elapsed}s\n")

    # Compute summaries per model
    summaries = {}
    for model in models:
        model_data = comparison["results"][model]
        summaries[model] = compute_summary({
            "criteria": criteria,
            "case_results": model_data["case_results"],
        })

    # Column widths
    crit_width = max(len(c["name"]) for c in criteria) + 2
    col_width = max(max(len(m) for m in models), 8)

    # Header row
    header = f"  {'Criteria':<{crit_width}}"
    for model in models:
        short = _short_model(model)
        header += f"  {short:^{col_width}}"
    print(header)
    print(f"  {'─' * crit_width}" + (f"  {'─' * col_width}" * len(models)))

    # Criteria rows
    for c in criteria:
        name = c["name"]
        row = f"  {name:<{crit_width}}"
        for model in models:
            avg = summaries[model]["criteria_averages"].get(name, 0.0)
            cell = f"{avg}/5"
            row += f"  {cell:^{col_width}}"
        print(row)

    # Weighted score row
    print(f"  {'─' * crit_width}" + (f"  {'─' * col_width}" * len(models)))
    row = f"  {'Weighted':<{crit_width}}"
    for model in models:
        ws = summaries[model]["weighted_score"]
        wp = summaries[model]["weighted_pct"]
        cell = f"{ws} ({wp}%)"
        row += f"  {cell:^{col_width}}"
    print(row)

    # Usage row
    print()
    for model in models:
        usage = comparison["results"][model].get("usage", {})
        if usage:
            short = _short_model(model)
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
            cost = usage.get("estimated_cost_usd", 0.0)
            print(f"  {short}: {tokens_in:,} in / {tokens_out:,} out  (~${cost:.4f})")

    print()


def _short_model(model: str) -> str:
    """Shorten a model name for display (e.g. 'claude-sonnet-4-6' → 'sonnet-4-6')."""
    return model.replace("claude-", "")


def save_report(report_data: dict, path: Path):
    """Save report data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report_data["summary"] = compute_summary(report_data)
    with open(path, "w") as f:
        json.dump(report_data, f, indent=2)


def save_comparison(comparison: dict, path: Path):
    """Save comparison data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
