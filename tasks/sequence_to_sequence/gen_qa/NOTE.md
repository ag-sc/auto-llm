# LLM-as-a-Judge criteria for gen-QA

The binary judge metrics in `utils.py` (via `auto_llm/evaluator/llm_judge.py`) and how they relate to RAG evaluation papers.

| Metric | Needs reference | Literature equivalent |
|---|---|---|
| `judge_faithfulness` | no | Faithfulness: RAGAS, ARES, RAGChecker |
| `judge_relevance` | no | Answer relevance: RAGAS, ARES |
| `judge_correctness` | yes | Claim precision vs. ground truth: RAGChecker |
| `judge_completeness` | yes | Claim recall vs. ground truth: RAGChecker |
| `judge_fluency` | no | General text-quality evaluation (e.g. G-Eval), not RAG-specific |
| `judge_conciseness` | no | Mostly from practitioner tooling, not papers |

## Prompt
All metrics come from **one** judge call per sample: the criteria are listed together and the judge returns a reason and a 0/1 verdict for each. So there is no separate prompt per metric; each metric is defined by its line in the `### Criteria` block. The text lives in `LLMJudge` (`REFERENCE_CRITERIA`, `REFERENCE_FREE_CRITERIA`, `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`).

| Metric | Criterion definition given to the judge |
|---|---|
| `judge_correctness` | The answer is factually consistent with the reference answer and does not contradict it. |
| `judge_completeness` | The answer covers all key points of the reference answer that are needed to answer the question. |
| `judge_faithfulness` | Every claim in the answer is supported by the given context; nothing is hallucinated. |
| `judge_relevance` | The answer directly addresses the question that was asked. |
| `judge_fluency` | The answer is grammatical, coherent and easy to read. |
| `judge_conciseness` | The answer contains no unnecessary padding, repetition or off-topic content. |

System prompt:
```text
You are a strict and impartial evaluator of answers to questions. For each criterion, first give a one-sentence reason, then a binary verdict: 1 if the answer fully satisfies the criterion, otherwise 0. Respond only with a JSON object.
```

User prompt (with a reference answer):
```text
Evaluate the answer according to the criteria below.

### Criteria
- correctness: The answer is factually consistent with the reference answer and does not contradict it.
- completeness: The answer covers all key points of the reference answer that are needed to answer the question.
- faithfulness: Every claim in the answer is supported by the given context; nothing is hallucinated.
- relevance: The answer directly addresses the question that was asked.
- fluency: The answer is grammatical, coherent and easy to read.
- conciseness: The answer contains no unnecessary padding, repetition or off-topic content.

### Question and context
{question_context}

### Reference answer
{reference}

### Answer to evaluate
{prediction}

Return a JSON object with exactly these keys: "correctness", "completeness", "faithfulness", "relevance", "fluency", "conciseness".
Each value must be an object of the form {"reason": "<one sentence>", "verdict": 0 or 1}.
```

Without a reference answer, the `correctness` and `completeness` lines, the `### Reference answer` block and their JSON keys are left out, and those two metrics are `None`. `{question_context}` is the sample's `input_text` (question plus abstract), `{reference}` its `output_text`, and `{prediction}` the evaluated model's answer.

The request also sets `temperature: 0` and a JSON schema (vLLM guided decoding) that forces exactly these keys, with `reason` generated before `verdict`.

## Differences from the papers
- **Context relevance** (RAGAS, ARES) is not included. It scores the retriever, not the answer, and in `pubmed_gen_qa` the context is the given abstract rather than retrieved. Add it when evaluating a real retrieval pipeline.
- **Granularity:** RAGAS and RAGChecker judge each claim in the answer and report the fraction supported (a continuous score). We give one 0/1 verdict per answer, which is simpler and stricter: a single unsupported claim sets faithfulness to 0. A per-claim version would turn correctness and completeness into precision and recall.
- **Binary vs. graded scores:** the evidence is mixed. Practitioner reports favour binary for consistency, while one recent preprint finds the best agreement with human ratings on a 0–5 scale. Validate against a small human-labelled sample.

## References
- RAGAS (Es et al., EACL 2024): https://arxiv.org/abs/2309.15217
- ARES (Saad-Falcon et al., NAACL 2024): https://aclanthology.org/2024.naacl-long.20
- RAGChecker (Ru et al., NeurIPS 2024): https://arxiv.org/abs/2408.08067
- Grading Scale Impact on LLM-as-a-Judge (preprint): https://arxiv.org/abs/2601.03444
