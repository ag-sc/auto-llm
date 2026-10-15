from typing import Dict, Any, List

import evaluate


BLEU = evaluate.load("bleu")
ROUGE = evaluate.load("rouge")


def process_results(doc: Dict[str, Any], result: List[str]):
    """
    Function to compute the metrics given the input and the generated response.

    :param doc: this includes the input
    :param result: this is the generated response
    :return: dict of metric key-value pairs
    """

    expected_response = [doc["output_text"]]
    predicted_response = [result[0]]

    bleu_scores = BLEU.compute(
        predictions=predicted_response, references=expected_response
    )

    rouge_scores = ROUGE.compute(
        predictions=predicted_response, references=expected_response
    )

    return {
        "bleu": bleu_scores.get("bleu"),
        "rouge1": rouge_scores.get("rouge1"),
        "rouge2": rouge_scores.get("rouge2"),
        "rougeL": rouge_scores.get("rougeL"),
    }
