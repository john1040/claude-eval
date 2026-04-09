"""Tests for the judge and report modules."""

import json
import pytest

from claude_eval.judge import parse_judge_response, build_judge_prompt
from claude_eval.report import compute_summary


class TestParseJudgeResponse:
    def test_clean_json(self):
        text = '{"scores": {"accuracy": {"score": 4, "reason": "Good"}}}'
        result = parse_judge_response(text)
        assert result["scores"]["accuracy"]["score"] == 4

    def test_json_with_markdown_fences(self):
        text = '```json\n{"scores": {"accuracy": {"score": 5, "reason": "Great"}}}\n```'
        result = parse_judge_response(text)
        assert result["scores"]["accuracy"]["score"] == 5

    def test_json_with_surrounding_text(self):
        text = 'Here are the results:\n{"scores": {"accuracy": {"score": 3, "reason": "OK"}}}\nDone.'
        result = parse_judge_response(text)
        assert result["scores"]["accuracy"]["score"] == 3

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_judge_response("This is not JSON at all")


class TestBuildJudgePrompt:
    def test_includes_criteria(self):
        criteria = [
            {"name": "accuracy", "description": "Is it accurate?", "weight": 0.5},
            {"name": "tone", "description": "Is the tone right?", "weight": 0.5},
        ]
        prompt = build_judge_prompt(
            system_prompt="Be helpful",
            user_input="What is Python?",
            response="Python is a programming language.",
            criteria=criteria,
        )
        assert "accuracy" in prompt
        assert "tone" in prompt
        assert "Python is a programming language" in prompt

    def test_includes_expected_keywords(self):
        criteria = [{"name": "accuracy", "description": "Is it accurate?", "weight": 1.0}]
        prompt = build_judge_prompt(
            system_prompt="Summarize",
            user_input="Some text",
            response="A summary",
            criteria=criteria,
            expected_keywords=["keyword1", "keyword2"],
        )
        assert "keyword1" in prompt
        assert "keyword2" in prompt


class TestComputeSummary:
    def test_basic_summary(self):
        report_data = {
            "criteria": [
                {"name": "accuracy", "weight": 0.5},
                {"name": "tone", "weight": 0.5},
            ],
            "case_results": [
                {
                    "scores": {
                        "accuracy": {"score": 4, "reason": "Good"},
                        "tone": {"score": 5, "reason": "Great"},
                    }
                },
                {
                    "scores": {
                        "accuracy": {"score": 4, "reason": "Good"},
                        "tone": {"score": 3, "reason": "OK"},
                    }
                },
            ],
        }
        summary = compute_summary(report_data)
        assert summary["criteria_averages"]["accuracy"] == 4.0
        assert summary["criteria_averages"]["tone"] == 4.0
        assert summary["weighted_score"] == 4.0
        assert summary["weighted_pct"] == 80.0

    def test_empty_results(self):
        report_data = {
            "criteria": [{"name": "accuracy", "weight": 1.0}],
            "case_results": [],
        }
        summary = compute_summary(report_data)
        assert summary["weighted_score"] == 0.0
