"""
LLM-as-a-Judge client for generative tasks.

The judge model is served separately (e.g. ``scripts/serve_llm_judge.sbatch`` with vLLM)
behind an OpenAI-compatible API. This module only sends HTTP requests to it, so it can be
called from lm-eval's ``process_results`` without loading a second model on the evaluator GPU.

Configuration (env vars, since ``process_results`` only receives ``(doc, result)``):
    LLM_JUDGE_BASE_URL     e.g. http://gpu-node:8000/v1. If unset, all judge metrics are None.
    LLM_JUDGE_MODEL        served model name (default: Qwen/Qwen2.5-7B-Instruct)
    LLM_JUDGE_API_KEY      optional, must match ``vllm serve --api-key``
    LLM_JUDGE_TIMEOUT      request timeout in seconds (default: 120)
    LLM_JUDGE_MAX_RETRIES  retries on connection errors / 5xx (default: 3)
"""

import ast
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Criteria that need a reference answer. Skipped (None) when the reference is missing.
REFERENCE_CRITERIA = {
    "correctness": "The answer is factually consistent with the reference answer and does not contradict it.",
    "completeness": "The answer covers all key points of the reference answer that are needed to answer the question.",
}
# Criteria that only need the question/context and the answer.
REFERENCE_FREE_CRITERIA = {
    "faithfulness": "Every claim in the answer is supported by the given context; nothing is hallucinated.",
    "relevance": "The answer directly addresses the question that was asked.",
    "fluency": "The answer is grammatical, coherent and easy to read.",
    "conciseness": "The answer contains no unnecessary padding, repetition or off-topic content.",
}
ALL_CRITERIA = list(REFERENCE_CRITERIA) + list(REFERENCE_FREE_CRITERIA)
METRIC_PREFIX = "judge_"

SYSTEM_PROMPT = (
    "You are a strict and impartial evaluator of answers to questions. "
    "For each criterion, first give a one-sentence reason, then a binary verdict: "
    "1 if the answer fully satisfies the criterion, otherwise 0. "
    "Respond only with a JSON object."
)

USER_PROMPT_TEMPLATE = """Evaluate the answer according to the criteria below.

### Criteria
{criteria}

### Question and context
{question_context}
{reference_block}
### Answer to evaluate
{prediction}

Return a JSON object with exactly these keys: {keys}.
Each value must be an object of the form {{"reason": "<one sentence>", "verdict": 0 or 1}}."""

REFERENCE_BLOCK_TEMPLATE = """
### Reference answer
{reference}
"""


def _criteria_for(has_reference: bool) -> Dict[str, str]:
    if has_reference:
        return {**REFERENCE_CRITERIA, **REFERENCE_FREE_CRITERIA}
    return dict(REFERENCE_FREE_CRITERIA)


def build_messages(question_context: str, prediction: str, reference: Optional[str]) -> List[Dict[str, str]]:
    criteria = _criteria_for(has_reference=reference is not None)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        criteria="\n".join(f"- {name}: {definition}" for name, definition in criteria.items()),
        question_context=question_context,
        reference_block=REFERENCE_BLOCK_TEMPLATE.format(reference=reference) if reference is not None else "",
        prediction=prediction,
        keys=", ".join(f'"{name}"' for name in criteria),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_json_schema(has_reference: bool) -> Dict:
    """JSON schema for guided decoding (vLLM ``response_format``). ``reason`` precedes ``verdict``."""
    verdict_schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "verdict": {"type": "integer", "enum": [0, 1]},
        },
        "required": ["reason", "verdict"],
    }
    criteria = list(_criteria_for(has_reference=has_reference))
    return {
        "type": "object",
        "properties": {name: verdict_schema for name in criteria},
        "required": criteria,
    }


def clean_and_extract_json(text: str) -> str:
    """Strips a (possibly unclosed) ```json fence. See ``tasks/sequence_to_structured_output/ner/utils.py``."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"```(?:json)?\s*(.*?)$", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def parse_dict(text: str) -> Union[Dict, None]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, SyntaxError):
            return None


def _to_binary(value) -> Optional[int]:
    """Accepts ``{"verdict": 1}``, ``1``, ``"1"``, ``True``, ``"yes"``... and returns 0/1 or None."""
    if isinstance(value, dict):
        value = value.get("verdict")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("1", "yes", "true"):
            return 1
        if value in ("0", "no", "false"):
            return 0
    return None


def parse_verdicts(text: str, has_reference: bool) -> Dict[str, Optional[int]]:
    """Maps the judge response to ``{"judge_<criterion>": 0 | 1 | None}`` for all criteria."""
    parsed = parse_dict(clean_and_extract_json(text or ""))
    if parsed is None:
        logger.warning("LLM judge: could not parse response: %s", (text or "")[:200])
        parsed = {}

    criteria = _criteria_for(has_reference=has_reference)
    return {
        f"{METRIC_PREFIX}{name}": _to_binary(parsed.get(name)) if name in criteria else None
        for name in ALL_CRITERIA
    }


def empty_verdicts() -> Dict[str, None]:
    return {f"{METRIC_PREFIX}{name}": None for name in ALL_CRITERIA}


class LLMJudge:
    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_JUDGE_MODEL,
        api_key: Optional[str] = None,
        timeout: float = 120,
        max_retries: int = 3,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def from_env(cls) -> Optional["LLMJudge"]:
        base_url = os.environ.get("LLM_JUDGE_BASE_URL")
        if not base_url:
            return None
        return cls(
            base_url=base_url,
            model=os.environ.get("LLM_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
            api_key=os.environ.get("LLM_JUDGE_API_KEY"),
            timeout=float(os.environ.get("LLM_JUDGE_TIMEOUT", 120)),
            max_retries=int(os.environ.get("LLM_JUDGE_MAX_RETRIES", 3)),
        )

    def _post(self, payload: Dict) -> Optional[str]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(self.url, json=payload, timeout=self.timeout)
                if response.status_code < 500:
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
                logger.warning("LLM judge: HTTP %s (attempt %d)", response.status_code, attempt + 1)
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning("LLM judge: %s (attempt %d)", e, attempt + 1)
            except (requests.HTTPError, KeyError, IndexError, ValueError) as e:
                # 4xx or malformed body: retrying will not help
                logger.warning("LLM judge: request failed: %s", e)
                return None
            if attempt < self.max_retries:
                time.sleep(2**attempt)
        return None

    def judge(self, question_context: str, prediction: str, reference: Optional[str] = None) -> Dict[str, Optional[int]]:
        has_reference = reference is not None
        payload = {
            "model": self.model,
            "messages": build_messages(question_context, prediction, reference),
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "judge_verdicts", "schema": build_json_schema(has_reference)},
            },
        }
        content = self._post(payload)
        if content is None:
            return empty_verdicts()
        return parse_verdicts(content, has_reference=has_reference)


_JUDGE: Optional[LLMJudge] = None
_JUDGE_INITIALIZED = False


def get_judge() -> Optional[LLMJudge]:
    """Lazily creates the judge from env vars. Returns None (and warns once) if not configured."""
    global _JUDGE, _JUDGE_INITIALIZED
    if not _JUDGE_INITIALIZED:
        _JUDGE = LLMJudge.from_env()
        _JUDGE_INITIALIZED = True
        if _JUDGE is None:
            logger.warning("LLM_JUDGE_BASE_URL is not set; LLM-as-a-Judge metrics are skipped.")
        else:
            logger.info("LLM judge: %s at %s", _JUDGE.model, _JUDGE.url)
    return _JUDGE


def is_enabled() -> bool:
    """True if a judge endpoint is configured (``LLM_JUDGE_BASE_URL`` is set)."""
    return get_judge() is not None


def judge(question_context: str, prediction: str, reference: Optional[str] = None) -> Dict[str, Optional[int]]:
    llm_judge = get_judge()
    if llm_judge is None:
        return empty_verdicts()
    return llm_judge.judge(question_context, prediction, reference)


def nan_safe_mean(items: List[Optional[float]]) -> Optional[float]:
    """lm-eval aggregation that ignores None (skipped / failed) values."""
    values = [x for x in items if x is not None]
    if not values:
        return None
    return sum(values) / len(values)
