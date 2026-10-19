from typing import Dict, Any, List

import evaluate

from auto_llm.evaluator.llm_judge import LLMJudge

# used as aggregation in the task YAML (``!function utils.nan_safe_mean``)
nan_safe_mean = LLMJudge.nan_safe_mean


BLEU = evaluate.load("bleu")
ROUGE = evaluate.load("rouge")


def process_results(doc: Dict[str, Any], result: List[str]):
    """
    Function to compute the metrics given the input and the generated response.
    Reference-based metrics are None when the doc has no reference answer.
    LLM-as-a-Judge metrics are only returned when a judge endpoint is configured
    (``llm_judge.base_url`` in the evaluator YAML); otherwise they are omitted from the results.

    :param doc: this includes the input
    :param result: this is the generated response
    :return: dict of metric key-value pairs
    """

    reference = (doc.get("output_text") or "").strip() or None
    prediction = result[0]

    metrics = {"bleu": None, "rouge1": None, "rouge2": None, "rougeL": None}
    if reference is not None:
        expected_response = [reference]
        predicted_response = [prediction]

        bleu_scores = BLEU.compute(
            predictions=predicted_response, references=expected_response
        )

        rouge_scores = ROUGE.compute(
            predictions=predicted_response, references=expected_response
        )

        metrics = {
            "bleu": bleu_scores.get("bleu"),
            "rouge1": rouge_scores.get("rouge1"),
            "rouge2": rouge_scores.get("rouge2"),
            "rougeL": rouge_scores.get("rougeL"),
        }

    if LLMJudge.is_enabled():
        metrics.update(
            LLMJudge.get().judge(
                question_context=doc["input_text"],
                prediction=prediction,
                reference=reference,
            )
        )
    return metrics
