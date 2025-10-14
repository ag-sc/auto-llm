from typing import Dict, Any, List

# TODO: Note the following
#  This script is not used. Only using log probs directly from lm-eval-harness.
#  Similar to https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/pubmedqa/pubmedqa.yaml
#  See https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/squadv2
#  for using both continuation and log probs


def process_results(doc: Dict[str, Any], result: List[str]):
    """
    Function to compute the metrics given the input and the generated response.

    :param doc: this includes the input
    :param result: this is the generated response
    :return: dict of metric key-value pairs
    """

    expected_response = [doc["output_text"]]
    predicted_response = [result[0]]

    print("expected response:", expected_response)
    print("predicted response:", predicted_response)

    acc = expected_response == predicted_response

    return {"acc": acc}
