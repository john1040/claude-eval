"""Judge: uses Claude as an LLM-as-judge to score eval responses."""

import asyncio
import json
import re

from claude_eval.runner import get_async_client


JUDGE_SYSTEM_PROMPT = """\
You are an impartial evaluation judge. You will be given:
1. A task description (system prompt)
2. A user input
3. A model response
4. Evaluation criteria

Score each criterion from 1-5 where:
  1 = Very poor
  2 = Poor
  3 = Adequate
  4 = Good
  5 = Excellent

You MUST respond with ONLY valid JSON in this exact format, no other text:
{
  "scores": {
    "<criterion_name>": {
      "score": <1-5>,
      "reason": "<brief explanation>"
    }
  }
}
"""


def build_judge_prompt(
    system_prompt: str,
    user_input: str,
    response: str,
    criteria: list[dict],
    expected_keywords: list[str] | None = None,
) -> str:
    """Build the user message for the judge."""
    criteria_desc = "\n".join(
        f"- {c['name']}: {c['description']}" for c in criteria
    )

    # XML tags prevent content inside one field from injecting instructions
    # into another section (prompt injection via crafted model responses).
    prompt = f"""<task_description>
{system_prompt}
</task_description>

<user_input>
{user_input}
</user_input>

<model_response>
{response}
</model_response>

<evaluation_criteria>
{criteria_desc}
</evaluation_criteria>
"""

    if expected_keywords:
        kw_str = ", ".join(f'"{k}"' for k in expected_keywords)
        prompt += f"\n<expected_keywords>{kw_str}</expected_keywords>\n"

    prompt += "\nScore each criterion 1-5. Respond with ONLY valid JSON."
    return prompt


def parse_judge_response(text: str) -> dict:
    """Parse the judge's JSON response, handling markdown fences."""
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse judge response as JSON:\n{text}")


async def _judge_single_async(
    client,
    judge_model: str,
    system_prompt: str,
    user_input: str,
    response: str,
    criteria: list[dict],
    expected_keywords: list[str] | None = None,
) -> dict:
    """Judge one case asynchronously. Returns parsed scores dict."""
    judge_prompt = build_judge_prompt(
        system_prompt=system_prompt,
        user_input=user_input,
        response=response,
        criteria=criteria,
        expected_keywords=expected_keywords,
    )
    message = await client.messages.create(
        model=judge_model,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    judge_text = "".join(
        block.text for block in message.content if block.type == "text"
    )
    return parse_judge_response(judge_text)


async def judge_results_async(
    judge_model: str,
    system_prompt: str,
    criteria: list[dict],
    cases: list[dict],
    responses: list[str],
) -> list[dict]:
    """Judge all eval responses in parallel."""
    client = get_async_client()
    tasks = [
        _judge_single_async(
            client=client,
            judge_model=judge_model,
            system_prompt=system_prompt,
            user_input=case["input"],
            response=response,
            criteria=criteria,
            expected_keywords=case.get("expected_keywords"),
        )
        for case, response in zip(cases, responses)
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for case, response, raw in zip(cases, responses, raw_results):
        if isinstance(raw, Exception):
            scores = {
                c["name"]: {"score": 0, "reason": f"Judge error: {raw}"}
                for c in criteria
            }
        else:
            scores = raw.get("scores", {})

        results.append({
            "input": case["input"].strip()[:200],
            "expected_keywords": case.get("expected_keywords", []),
            "response": response.strip()[:500],
            "scores": scores,
        })

    return results


def judge_results(
    judge_model: str,
    system_prompt: str,
    criteria: list[dict],
    cases: list[dict],
    responses: list[str],
) -> list[dict]:
    """Synchronous wrapper around judge_results_async."""
    return asyncio.run(
        judge_results_async(judge_model, system_prompt, criteria, cases, responses)
    )
