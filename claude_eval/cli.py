"""CLI entry point for claude-eval."""

import asyncio
import json
import time
from pathlib import Path

import click
import yaml

from claude_eval.runner import run_eval_async
from claude_eval.judge import judge_results_async
from claude_eval.report import print_report, print_comparison_report, save_report, save_comparison


_REQUIRED_FIELDS = ["name", "model", "system_prompt", "cases", "judge"]
_REQUIRED_JUDGE_FIELDS = ["criteria"]


def _load_eval(eval_file: Path) -> dict:
    with open(eval_file) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise click.ClickException(f"{eval_file}: YAML must be a mapping, got {type(data).__name__}")

    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise click.ClickException(f"{eval_file}: missing required fields: {', '.join(missing)}")

    missing_judge = [f for f in _REQUIRED_JUDGE_FIELDS if f not in data.get("judge", {})]
    if missing_judge:
        raise click.ClickException(f"{eval_file}: missing judge fields: {', '.join(missing_judge)}")

    if not isinstance(data["cases"], list) or len(data["cases"]) == 0:
        raise click.ClickException(f"{eval_file}: 'cases' must be a non-empty list")

    for i, case in enumerate(data["cases"]):
        if not isinstance(case, dict) or "input" not in case:
            raise click.ClickException(f"{eval_file}: case {i+1} must have an 'input' field")

    return data


@click.group()
@click.version_option(version="0.2.0")
def main():
    """claude-eval: Evaluate and compare LLM outputs with structured criteria."""
    pass


@main.command()
@click.argument("eval_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Save JSON results to this path.")
@click.option("--verbose", "-v", is_flag=True, help="Show per-case responses.")
@click.option("--judge-model", default=None,
              help="Model to use as judge (defaults to same model being evaluated).")
def run(eval_file: Path, output: Path | None, verbose: bool, judge_model: str | None):
    """Run an evaluation from a YAML definition file."""
    eval_def = _load_eval(eval_file)

    eval_name = eval_def.get("name", eval_file.stem)
    model = eval_def["model"]
    system_prompt = eval_def["system_prompt"]
    cases = eval_def["cases"]
    criteria = eval_def["judge"]["criteria"]
    judge_model = judge_model or model

    click.echo(f"\n  Running eval: {click.style(eval_name, bold=True)}")
    click.echo(f"  Model:  {model}")
    click.echo(f"  Cases:  {len(cases)}")
    click.echo(f"  Judge:  {judge_model}\n")

    start = time.time()

    click.echo("  Running cases in parallel...")
    responses, usage = asyncio.run(
        run_eval_async(model=model, system_prompt=system_prompt, cases=cases)
    )

    if verbose:
        for i, resp in enumerate(responses):
            click.echo(f"\n  --- Case {i+1} ---")
            click.echo(f"  Input:    {cases[i]['input'][:80].strip()}...")
            click.echo(f"  Response: {resp[:120].strip()}...")

    click.echo("  Scoring with LLM-as-judge in parallel...")
    scored = asyncio.run(
        judge_results_async(
            judge_model=judge_model,
            system_prompt=system_prompt,
            criteria=criteria,
            cases=cases,
            responses=responses,
        )
    )

    elapsed = round(time.time() - start, 1)

    report_data = {
        "eval_name": eval_name,
        "model": model,
        "judge_model": judge_model,
        "num_cases": len(cases),
        "elapsed_seconds": elapsed,
        "criteria": criteria,
        "usage": usage,
        "case_results": scored,
    }

    print_report(report_data)

    if output:
        save_report(report_data, output)
        click.echo(f"  Results saved to {output}\n")


@main.command()
@click.argument("eval_file", type=click.Path(exists=True, path_type=Path))
@click.option("--models", "-m", required=True,
              help="Comma-separated list of models to compare, e.g. claude-sonnet-4-6,claude-haiku-4-5-20251001")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Save JSON results to this path.")
@click.option("--judge-model", default=None,
              help="Model to use as judge (defaults to first model in --models).")
def compare(eval_file: Path, models: str, output: Path | None, judge_model: str | None):
    """Compare multiple models on the same eval, side-by-side."""
    eval_def = _load_eval(eval_file)

    eval_name = eval_def.get("name", eval_file.stem)
    system_prompt = eval_def["system_prompt"]
    cases = eval_def["cases"]
    criteria = eval_def["judge"]["criteria"]
    model_list = [m.strip() for m in models.split(",")]
    judge = judge_model or model_list[0]

    click.echo(f"\n  Comparing: {click.style(eval_name, bold=True)}")
    click.echo(f"  Models:  {', '.join(model_list)}")
    click.echo(f"  Cases:   {len(cases)}")
    click.echo(f"  Judge:   {judge}\n")

    start = time.time()

    # Run all models in parallel
    click.echo("  Running all models in parallel...")

    async def run_all():
        tasks = {
            model: run_eval_async(model=model, system_prompt=system_prompt, cases=cases)
            for model in model_list
        }
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))

    model_outputs = asyncio.run(run_all())

    # Judge all models in parallel
    click.echo("  Scoring all models in parallel...")

    async def judge_all():
        tasks = {
            model: judge_results_async(
                judge_model=judge,
                system_prompt=system_prompt,
                criteria=criteria,
                cases=cases,
                responses=model_outputs[model][0],
            )
            for model in model_list
        }
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))

    scored_by_model = asyncio.run(judge_all())

    elapsed = round(time.time() - start, 1)

    comparison = {
        "eval_name": eval_name,
        "models": model_list,
        "judge_model": judge,
        "num_cases": len(cases),
        "elapsed_seconds": elapsed,
        "criteria": criteria,
        "results": {
            model: {
                "usage": model_outputs[model][1],
                "case_results": scored_by_model[model],
            }
            for model in model_list
        },
    }

    print_comparison_report(comparison)

    if output:
        save_comparison(comparison, output)
        click.echo(f"  Results saved to {output}\n")


@main.command()
@click.argument("results_file", type=click.Path(exists=True, path_type=Path))
def report(results_file: Path):
    """Print a report from a saved JSON results file."""
    with open(results_file) as f:
        data = json.load(f)

    if "models" in data:
        print_comparison_report(data)
    else:
        print_report(data)


if __name__ == "__main__":
    main()
