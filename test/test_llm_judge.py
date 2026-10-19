import json
from unittest import mock

import pytest
import requests

from auto_llm.evaluator import llm_judge
from auto_llm.evaluator.llm_judge import LLMJudge, nan_safe_mean, parse_verdicts

FREE = ["faithfulness", "relevance", "fluency", "conciseness"]
ALL = ["correctness", "completeness"] + FREE


def _verdicts(criteria, verdict=1):
    return {c: {"reason": "ok", "verdict": verdict} for c in criteria}


def _response(status_code=200, content=None):
    response = mock.Mock(status_code=status_code)
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    response.raise_for_status.side_effect = requests.HTTPError() if status_code >= 400 else None
    return response


@pytest.fixture
def judge():
    return LLMJudge(base_url="http://judge:8000/v1/", model="judge-model", max_retries=2)


def test_parse_verdicts_with_reference():
    scores = parse_verdicts(json.dumps(_verdicts(ALL)), has_reference=True)
    assert scores == {f"judge_{c}": 1 for c in ALL}


def test_parse_verdicts_without_reference_skips_reference_criteria():
    scores = parse_verdicts(json.dumps(_verdicts(ALL, verdict=0)), has_reference=False)
    assert scores["judge_correctness"] is None
    assert scores["judge_completeness"] is None
    assert all(scores[f"judge_{c}"] == 0 for c in FREE)


def test_parse_verdicts_fenced_and_loose_values():
    text = '```json\n{"faithfulness": 1, "relevance": "yes", "fluency": {"verdict": "0"}, "conciseness": 5}\n```'
    scores = parse_verdicts(text, has_reference=False)
    assert scores["judge_faithfulness"] == 1
    assert scores["judge_relevance"] == 1
    assert scores["judge_fluency"] == 0
    assert scores["judge_conciseness"] is None


def test_parse_verdicts_malformed():
    scores = parse_verdicts("not json at all", has_reference=True)
    assert set(scores) == {f"judge_{c}" for c in ALL}
    assert all(v is None for v in scores.values())


def test_judge_payload_with_reference(judge):
    with mock.patch.object(judge.session, "post", return_value=_response(content=json.dumps(_verdicts(ALL)))) as post:
        scores = judge.judge("Question: q", "answer", reference="ref")

    assert post.call_args.args[0] == "http://judge:8000/v1/chat/completions"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "judge-model"
    assert payload["temperature"] == 0
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ALL
    assert "ref" in payload["messages"][1]["content"]
    assert all(v == 1 for v in scores.values())


def test_judge_payload_without_reference(judge):
    with mock.patch.object(judge.session, "post", return_value=_response(content=json.dumps(_verdicts(FREE)))) as post:
        scores = judge.judge("Question: q", "answer")

    payload = post.call_args.kwargs["json"]
    assert payload["response_format"]["json_schema"]["schema"]["required"] == FREE
    assert "Reference answer" not in payload["messages"][1]["content"]
    assert scores["judge_correctness"] is None
    assert scores["judge_faithfulness"] == 1


@mock.patch("auto_llm.evaluator.llm_judge.time.sleep")
def test_judge_retries_on_5xx(_sleep, judge):
    responses = [_response(status_code=503), _response(content=json.dumps(_verdicts(FREE)))]
    with mock.patch.object(judge.session, "post", side_effect=responses) as post:
        scores = judge.judge("q", "a")
    assert post.call_count == 2
    assert scores["judge_relevance"] == 1


@mock.patch("auto_llm.evaluator.llm_judge.time.sleep")
def test_judge_gives_up_on_connection_errors(_sleep, judge):
    with mock.patch.object(judge.session, "post", side_effect=requests.ConnectionError("down")) as post:
        scores = judge.judge("q", "a", reference="r")
    assert post.call_count == 3
    assert all(v is None for v in scores.values())


def test_judge_does_not_retry_on_4xx(judge):
    with mock.patch.object(judge.session, "post", return_value=_response(status_code=400)) as post:
        scores = judge.judge("q", "a")
    assert post.call_count == 1
    assert all(v is None for v in scores.values())


def test_module_judge_without_base_url(monkeypatch):
    monkeypatch.delenv("LLM_JUDGE_BASE_URL", raising=False)
    monkeypatch.setattr(llm_judge, "_JUDGE", None)
    monkeypatch.setattr(llm_judge, "_JUDGE_INITIALIZED", False)
    scores = llm_judge.judge("q", "a", reference="r")
    assert set(scores) == {f"judge_{c}" for c in ALL}
    assert all(v is None for v in scores.values())


def test_nan_safe_mean():
    assert nan_safe_mean([1, 0, None, 1]) == pytest.approx(2 / 3)
    assert nan_safe_mean([None, None]) is None
    assert nan_safe_mean([]) is None


def test_is_enabled(monkeypatch):
    for base_url, expected in [(None, False), ("http://judge:8000/v1", True)]:
        if base_url is None:
            monkeypatch.delenv("LLM_JUDGE_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("LLM_JUDGE_BASE_URL", base_url)
        monkeypatch.setattr(llm_judge, "_JUDGE", None)
        monkeypatch.setattr(llm_judge, "_JUDGE_INITIALIZED", False)
        assert llm_judge.is_enabled() is expected
