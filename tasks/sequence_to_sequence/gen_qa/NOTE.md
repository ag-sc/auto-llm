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

## Differences from the papers
- **Context relevance** (RAGAS, ARES) is not included. It scores the retriever, not the answer, and in `pubmed_gen_qa` the context is the given abstract rather than retrieved. Add it when evaluating a real retrieval pipeline.
- **Granularity:** RAGAS and RAGChecker judge each claim in the answer and report the fraction supported (a continuous score). We give one 0/1 verdict per answer, which is simpler and stricter: a single unsupported claim sets faithfulness to 0. A per-claim version would turn correctness and completeness into precision and recall.
- **Binary vs. graded scores:** the evidence is mixed. Practitioner reports favour binary for consistency, while one recent preprint finds the best agreement with human ratings on a 0–5 scale. Validate against a small human-labelled sample.

## References
- RAGAS (Es et al., EACL 2024): https://arxiv.org/abs/2309.15217
- ARES (Saad-Falcon et al., NAACL 2024): https://aclanthology.org/2024.naacl-long.20
- RAGChecker (Ru et al., NeurIPS 2024): https://arxiv.org/abs/2408.08067
- Grading Scale Impact on LLM-as-a-Judge (preprint): https://arxiv.org/abs/2601.03444
