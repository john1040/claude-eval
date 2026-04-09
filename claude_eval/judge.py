"""Judge: uses Claude as an LLM-as-judge to score eval responses."""

import json
import re


def _get_client():
    from claude_eval.runner import get_client
    return get_client()


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

    prompt = f"""## Task Description (System Prompt Being Evaluated)
{system_prompt}

## User Input
{user_input}

## Model Response
{response}

## Evaluation Criteria
{criteria_desc}
"""

    if expected_keywords:
        kw_str = ", ".join(f'"{k}"' for k in expected_keywords)
        prompt += f"\n## Expected Keywords\nThe response should mention: {kw_str}\n"

    prompt += "\nScore each criterion 1-5. Respond with ONLY valid JSON."
    return prompt


def parse_judge_response(text: str) -> dict:
    """Parse the judge's JSON response, handling markdown fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to find JSON object in the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse judge response as JSON:\n{text}")


def judge_single(
    client,
    model: str,
    system_prompt: str,
    user_input: str,
    response: str,
    criteria: list[dict],
    expected_keywords: list[str] | None = None,
) -> dict:
    """Judge a single case. Returns parsed scores dict."""
    judge_prompt = build_judge_prompt(
        system_prompt=system_prompt,
        user_input=user_input,
        response=response,
        criteria=criteria,
        expected_keywords=expected_keywords,
    )

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": judge_prompt}],
    )

    judge_text = "".join(
        block.text for block in message.content if block.type == "text"
    )
    return parse_judge_response(judge_text)


def judge_results(
    model: str,
    criteria: list[dict],
    cases: list[dict],
    responses: list[str],
) -> list[dict]:
    """Judge all eval responses and return scored results.

    Args:
        model: Claude model to use as judge.
        criteria: List of criterion dicts with name, description, weight.
        cases: Original eval cases (need input + expected_keywords).
        responses: Model responses to judge.

    Returns:
        List of dicts with case info, response, and scores.
    """
    client = _get_client()
    results = []

    # We need the system_prompt from the eval, but it's not passed here.
    # We'll reconstruct context from the case input.
    for case, response in zip(cases, responses):
        try:
            parsed = judge_single(
                client=client,
                model=model,
                system_prompt="(see user input for the original task)",
                user_input=case["input"],
                response=response,
                criteria=criteria,
                expected_keywords=case.get("expected_keywords"),
            )
            scores = parsed.get("scores", {})
        except (ValueError, json.JSONDecodeError) as e:
            # If judge fails to return valid JSON, record an error
            scores = {
                c["name"]: {"score": 0, "reason": f"Judge error: {e}"}
                for c in criteria
            }

        results.append({
            "input": case["input"].strip()[:200],
            "expected_keywords": case.get("expected_keywords", []),
            "response": response.strip()[:500],
            "scores": scores,
        })

    return results
