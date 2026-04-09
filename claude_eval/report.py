"""Report: format and display evaluation results."""

import json
from pathlib import Path


def compute_summary(report_data: dict) -> dict:
    """Compute aggregate scores from case results."""
    criteria = report_data["criteria"]
    case_results = report_data["case_results"]

    if not case_results:
        return {"criteria_averages": {}, "weighted_score": 0.0, "weighted_pct": 0.0}

    # Compute average score per criterion
    criteria_averages = {}
    for c in criteria:
        name = c["name"]
        scores = []
        for case in case_results:
            case_scores = case.get("scores", {})
            if name in case_scores and case_scores[name].get("score", 0) > 0:
                scores.append(case_scores[name]["score"])
        avg = sum(scores) / len(scores) if scores else 0.0
        criteria_averages[name] = round(avg, 1)

    # Weighted score
    total_weight = sum(c.get("weight", 1.0) for c in criteria)
    weighted_sum = sum(
        criteria_averages.get(c["name"], 0) * c.get("weight", 1.0)
        for c in criteria
    )
    weighted_score = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0
    weighted_pct = round((weighted_score / 5.0) * 100, 1)

    return {
        "criteria_averages": criteria_averages,
        "weighted_score": weighted_score,
        "weighted_pct": weighted_pct,
    }


def print_report(report_data: dict):
    """Print a formatted CLI report."""
    eval_name = report_data["eval_name"]
    model = report_data["model"]
    num_cases = report_data["num_cases"]
    elapsed = report_data["elapsed_seconds"]
    criteria = report_data["criteria"]

    summary = compute_summary(report_data)

    line = "═" * 50
    print(f"\n  {line}")
    print(f"  claude-eval results: {eval_name}")
    print(f"  {line}\n")
    print(f"  Model:  {model}")
    print(f"  Cases:  {num_cases}")
    print(f"  Time:   {elapsed}s\n")

    # Table header
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

    # Per-case breakdown
    for i, case_result in enumerate(report_data["case_results"]):
        scores_str = ", ".join(
            f"{k}={v.get('score', '?')}"
            for k, v in case_result.get("scores", {}).items()
        )
        print(f"  Case {i+1}: {scores_str}")

    print()


def save_report(report_data: dict, path: Path):
    """Save report data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add summary to saved data
    report_data["summary"] = compute_summary(report_data)

    with open(path, "w") as f:
        json.dump(report_data, f, indent=2)
