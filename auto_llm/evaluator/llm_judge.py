import ast
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    LLM-as-a-Judge client for generative tasks.

    The judge model is served separately (e.g. ``scripts/serve_llm_judge.sbatch`` with vLLM)
    behind an OpenAI-compatible API. This class only sends HTTP requests to it, so it can be
    called from lm-eval's ``process_results`` without loading a second model on the evaluator GPU.

    Configured only from the ``llm_judge`` section of the evaluator YAML (env vars are ignored).
    ``auto_llm/evaluator/utils.py`` passes that section to ``LLMJudge.configure`` before evaluation, and
    ``process_results`` (which only receives ``(doc, result)``) uses the shared judge from ``LLMJudge.get``:
        llm_judge:
          base_url: http://worker-15.GPU.CIT-EC.NET:8000/v1   # required; without it judge metrics are None
          model: Qwen/Qwen2.5-7B-Instruct                     # served model name (default shown)
          api_key: ...                                        # optional, must match ``vllm serve --api-key``
          timeout: 120                                        # request timeout in seconds
          max_retries: 3                                      # retries on connection errors / 5xx

    Every call to ``judge`` is kept in ``records`` (inputs, per-criterion reasons and verdicts), which
    ``save_records`` writes as JSONL and ``records_as_table`` turns into a W&B table.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    CONFIG_KEYS = ("base_url", "model", "api_key", "timeout", "max_retries")

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

    # Shared judge set by ``configure`` (``process_results`` cannot receive it as an argument).
    _instance: Optional["LLMJudge"] = None

    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        timeout: float = 120,
        max_retries: int = 3,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        # One entry per judged sample (inputs, per-criterion reasons and verdicts), see ``save_records``.
        self.records: List[Dict] = []
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    # ----- shared judge (configured from the evaluator YAML) -----

    @classmethod
    def configure(cls, judge_config: Optional[Dict]) -> None:
        """Creates the shared judge from the evaluator YAML's ``llm_judge`` section; disables it if absent."""
        judge_config = judge_config or {}
        unknown = set(judge_config) - set(cls.CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Unknown llm_judge config keys: {sorted(unknown)}. Allowed: {list(cls.CONFIG_KEYS)}")
        if not judge_config.get("base_url"):
            cls._instance = None
            logger.warning("No llm_judge.base_url in the evaluator config; LLM-as-a-Judge metrics are skipped.")
            return
        cls._instance = cls(**{key: value for key, value in judge_config.items() if value is not None})
        logger.info("LLM judge: %s at %s", cls._instance.model, cls._instance.url)

    @classmethod
    def get(cls) -> Optional["LLMJudge"]:
        """The shared judge set by ``configure``, or None if not configured."""
        return cls._instance

    @classmethod
    def is_enabled(cls) -> bool:
        """True if a judge endpoint is configured (``llm_judge.base_url`` in the evaluator YAML)."""
        return cls._instance is not None

    # ----- judging -----

    def judge(self, question_context: str, prediction: str, reference: Optional[str] = None) -> Dict[str, Optional[int]]:
        has_reference = reference is not None
        payload = {
            "model": self.model,
            "messages": self.build_messages(question_context, prediction, reference),
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "judge_verdicts", "schema": self.build_json_schema(has_reference)},
            },
        }
        content = self._post(payload)
        verdicts = self.empty_verdicts() if content is None else self.parse_verdicts(content, has_reference=has_reference)
        self.records.append(
            {
                "question_context": question_context,
                "prediction": prediction,
                "reference": reference,
                "verdicts": verdicts,
                "reasons": self.extract_reasons(content),
                "raw_response": content,
            }
        )
        return verdicts

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

    # ----- records -----

    def save_records(self, path: Union[str, Path]) -> Optional[Path]:
        """Writes the per-sample records (with reasons) as JSONL. Returns the path, or None if nothing was judged."""
        if not self.records:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved %d LLM judge records to %s", len(self.records), path)
        return path

    def records_as_table(self) -> Tuple[List[str], List[List]]:
        """The records as ``(columns, rows)`` for a W&B table: inputs, then verdict and reason per criterion."""
        columns = ["question_context", "prediction", "reference"]
        for name in self.ALL_CRITERIA:
            columns += [f"{self.METRIC_PREFIX}{name}", f"{self.METRIC_PREFIX}{name}_reason"]
        rows = []
        for record in self.records:
            row = [record["question_context"], record["prediction"], record["reference"]]
            for name in self.ALL_CRITERIA:
                key = f"{self.METRIC_PREFIX}{name}"
                row += [record["verdicts"].get(key), record["reasons"].get(key)]
            rows.append(row)
        return columns, rows

    # ----- prompt and response helpers -----

    @classmethod
    def criteria_for(cls, has_reference: bool) -> Dict[str, str]:
        if has_reference:
            return {**cls.REFERENCE_CRITERIA, **cls.REFERENCE_FREE_CRITERIA}
        return dict(cls.REFERENCE_FREE_CRITERIA)

    @classmethod
    def build_messages(cls, question_context: str, prediction: str, reference: Optional[str]) -> List[Dict[str, str]]:
        criteria = cls.criteria_for(has_reference=reference is not None)
        user_prompt = cls.USER_PROMPT_TEMPLATE.format(
            criteria="\n".join(f"- {name}: {definition}" for name, definition in criteria.items()),
            question_context=question_context,
            reference_block=cls.REFERENCE_BLOCK_TEMPLATE.format(reference=reference) if reference is not None else "",
            prediction=prediction,
            keys=", ".join(f'"{name}"' for name in criteria),
        )
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @classmethod
    def build_json_schema(cls, has_reference: bool) -> Dict:
        """JSON schema for guided decoding (vLLM ``response_format``). ``reason`` precedes ``verdict``."""
        verdict_schema = {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "verdict": {"type": "integer", "enum": [0, 1]},
            },
            "required": ["reason", "verdict"],
        }
        criteria = list(cls.criteria_for(has_reference=has_reference))
        return {
            "type": "object",
            "properties": {name: verdict_schema for name in criteria},
            "required": criteria,
        }

    @staticmethod
    def clean_and_extract_json(text: str) -> str:
        """Strips a (possibly unclosed) ```json fence. See ``tasks/sequence_to_structured_output/ner/utils.py``."""
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if not match:
            match = re.search(r"```(?:json)?\s*(.*?)$", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
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

    @staticmethod
    def to_binary(value) -> Optional[int]:
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

    @classmethod
    def parse_verdicts(cls, text: str, has_reference: bool) -> Dict[str, Optional[int]]:
        """Maps the judge response to ``{"judge_<criterion>": 0 | 1 | None}`` for all criteria."""
        parsed = cls.parse_dict(cls.clean_and_extract_json(text or ""))
        if parsed is None:
            logger.warning("LLM judge: could not parse response: %s", (text or "")[:200])
            parsed = {}

        criteria = cls.criteria_for(has_reference=has_reference)
        return {
            f"{cls.METRIC_PREFIX}{name}": cls.to_binary(parsed.get(name)) if name in criteria else None
            for name in cls.ALL_CRITERIA
        }

    @classmethod
    def extract_reasons(cls, content: Optional[str]) -> Dict[str, Optional[str]]:
        """Maps the judge response to ``{"judge_<criterion>": "<reason>"}`` for the criteria it answered."""
        parsed = cls.parse_dict(cls.clean_and_extract_json(content or "")) or {}
        return {
            f"{cls.METRIC_PREFIX}{name}": value.get("reason") if isinstance(value, dict) else None
            for name, value in parsed.items()
            if name in cls.ALL_CRITERIA
        }

    @classmethod
    def empty_verdicts(cls) -> Dict[str, None]:
        return {f"{cls.METRIC_PREFIX}{name}": None for name in cls.ALL_CRITERIA}

    @staticmethod
    def nan_safe_mean(items: List[Optional[float]]) -> Optional[float]:
        """lm-eval aggregation that ignores None (skipped / failed) values."""
        values = [x for x in items if x is not None]
        if not values:
            return None
        return sum(values) / len(values)
