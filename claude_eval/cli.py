"""CLI entry point for claude-eval."""

import json
import time
from pathlib import Path

import click
import yaml

from claude_eval.runner import run_eval
from claude_eval.judge import judge_results
from claude_eval.report import print_report, save_report


@click.group()
@click.version_option(version="0.1.0")
def main():
    """claude-eval: Evaluate LLM outputs with structured criteria."""
    pass


@main.command()
@click.argument("eval_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Save JSON results to this path.")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output per case.")
@click.option("--judge-model", default=None,
              help="Model to use as judge (defaults to same model being evaluated).")
def run(eval_file: Path, output: Path | None, verbose: bool, judge_model: str | None):
    """Run an evaluation from a YAML definition file."""
    # Load eval definition
    with open(eval_file) as f:
        eval_def = yaml.safe_load(f)

    eval_name = eval_def.get("name", eval_file.stem)
    model = eval_def["model"]
    system_prompt = eval_def["system_prompt"]
    cases = eval_def["cases"]
    judge_config = eval_def["judge"]
    judge_model = judge_model or model

    click.echo(f"\n  Running eval: {click.style(eval_name, bold=True)}")
    click.echo(f"  Model: {model}")
    click.echo(f"  Cases: {len(cases)}")
    click.echo(f"  Judge: {judge_model}")
    click.echo()

    start = time.time()

    # Step 1: Run all cases against Claude
    click.echo("  ⏳ Running cases against Claude API...")
    responses = run_eval(model=model, system_prompt=system_prompt, cases=cases)

    if verbose:
        for i, resp in enumerate(responses):
            click.echo(f"\n  --- Case {i+1} ---")
            click.echo(f"  Input:    {cases[i]['input'][:80].strip()}...")
            click.echo(f"  Response: {resp[:120].strip()}...")

    # Step 2: Judge each response
    click.echo("  ⏳ Scoring with LLM-as-judge...")
    scored = judge_results(
        model=judge_model,
        criteria=judge_config["criteria"],
        cases=cases,
        responses=responses,
    )

    elapsed = time.time() - start

    # Step 3: Report
    report_data = {
        "eval_name": eval_name,
        "model": model,
        "judge_model": judge_model,
        "num_cases": len(cases),
        "elapsed_seconds": round(elapsed, 1),
        "criteria": judge_config["criteria"],
        "case_results": scored,
    }

    print_report(report_data)

    if output:
        save_report(report_data, output)
        click.echo(f"\n  Results saved to {output}")


@main.command()
@click.argument("results_file", type=click.Path(exists=True, path_type=Path))
def report(results_file: Path):
    """Print a report from a saved JSON results file."""
    with open(results_file) as f:
        report_data = json.load(f)
    print_report(report_data)


if __name__ == "__main__":
    main()
