"""Runner: sends eval cases to the Claude API and collects responses."""

import asyncio
import os

# Max simultaneous API calls. Keeps costs predictable and avoids rate-limit
# errors on large evals. Raise if your API tier supports higher throughput.
MAX_CONCURRENT = 10


# Approximate cost per million tokens (input, output) in USD.
# Update these if Anthropic changes pricing.
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    "opus":   (15.00, 75.00),
    "sonnet": ( 3.00, 15.00),
    "haiku":  ( 0.80,  4.00),
}


def _cost_per_million(model: str) -> tuple[float, float]:
    """Return (input_$/M, output_$/M) for a model, matched by substring."""
    model_lower = model.lower()
    for key, costs in _MODEL_COSTS.items():
        if key in model_lower:
            return costs
    return (3.00, 15.00)  # default to sonnet pricing if unknown


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a given token count."""
    inp_rate, out_rate = _cost_per_million(model)
    return round((input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000, 6)


def get_client():
    """Create a synchronous Anthropic client."""
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is required.\n"
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    return Anthropic(api_key=api_key)


def get_async_client():
    """Create an async Anthropic client."""
    from anthropic import AsyncAnthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is required.\n"
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    return AsyncAnthropic(api_key=api_key)


async def _run_single_async(
    client,
    model: str,
    system_prompt: str,
    user_input: str,
    max_tokens: int = 1024,
) -> tuple[str, dict]:
    """Run one case. Returns (response_text, usage_dict)."""
    message = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }
    return text, usage


async def run_eval_async(
    model: str,
    system_prompt: str,
    cases: list[dict],
    max_tokens: int = 1024,
) -> tuple[list[str], dict]:
    """Run all eval cases in parallel (bounded by MAX_CONCURRENT).
    Returns (responses, aggregate_usage).
    """
    client = get_async_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def _guarded(case):
        async with semaphore:
            return await _run_single_async(client, model, system_prompt, case["input"], max_tokens)

    tasks = [_guarded(case) for case in cases]
    pairs = await asyncio.gather(*tasks)

    responses = [p[0] for p in pairs]
    total_input = sum(p[1]["input_tokens"] for p in pairs)
    total_output = sum(p[1]["output_tokens"] for p in pairs)
    usage = {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "estimated_cost_usd": estimate_cost(model, total_input, total_output),
    }
    return responses, usage


def run_eval(
    model: str,
    system_prompt: str,
    cases: list[dict],
    max_tokens: int = 1024,
) -> tuple[list[str], dict]:
    """Synchronous wrapper around run_eval_async."""
    return asyncio.run(run_eval_async(model, system_prompt, cases, max_tokens))
