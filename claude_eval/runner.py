"""Runner: sends eval cases to the Claude API and collects responses."""

import os


def get_client():
    """Create an Anthropic client. Raises if ANTHROPIC_API_KEY is not set."""
    from anthropic import Anthropic

    """Create an Anthropic client. Raises if ANTHROPIC_API_KEY is not set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is required.\n"
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    return Anthropic(api_key=api_key)


def run_single_case(
    client: Anthropic,
    model: str,
    system_prompt: str,
    user_input: str,
    max_tokens: int = 1024,
) -> str:
    """Run a single eval case and return the model's text response."""
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
    )
    # Extract text from content blocks
    return "".join(
        block.text for block in message.content if block.type == "text"
    )


def run_eval(
    model: str,
    system_prompt: str,
    cases: list[dict],
    max_tokens: int = 1024,
) -> list[str]:
    """Run all eval cases sequentially and return a list of responses.

    Args:
        model: Claude model identifier (e.g. 'claude-sonnet-4-20250514').
        system_prompt: The system prompt being evaluated.
        cases: List of case dicts, each with an 'input' key.
        max_tokens: Max tokens per response.

    Returns:
        List of response strings, one per case.
    """
    client = get_client()
    responses = []

    for case in cases:
        user_input = case["input"]
        response = run_single_case(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_input=user_input,
            max_tokens=max_tokens,
        )
        responses.append(response)

    return responses
