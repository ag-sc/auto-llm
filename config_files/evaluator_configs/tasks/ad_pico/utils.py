import json
import re
from typing import Dict, Any, List, Union


def process_results(doc: Dict[str, Any], result: List[str]):
    """
    Function to compute the metrics given the input and the generated response.

    :param doc: this includes the input
    :param result: this is the generated response
    :return: dict of metric key-value pairs
    """

    print("result\n", result)

    expected_entities_dict = doc["entities"]
    predicted_response = result[0]

    predicted_response = extract_from_json_tags(text=predicted_response)
    predicted_response_dict = parse_dict(text=predicted_response)

    print("expected_entities_dict\n", expected_entities_dict)
    print("predicted_response_dict\n", predicted_response_dict)

    full_match_score = 0
    if not isinstance(predicted_response_dict, dict):
        return {"exact_match": full_match_score}

    for key, expected_value in expected_entities_dict.items():
        predicted_value = predicted_response_dict[key]
        full_match = set(expected_value) == set(predicted_value)
        print(
            f"Key: {key}, Expected: {expected_value}, Predicted: {predicted_value}, Full Match: {full_match}"
        )
        full_match_score += full_match

    full_match_score /= len(expected_entities_dict)

    print("full_match_score", full_match_score)
    print("----------------------------")

    return {"exact_match": full_match_score}


def parse_dict(text: str) -> Union[Dict | str]:
    try:
        data = json.loads(text)
        return data
    except:
        print(f"Can't parse JSON: {text}")
        return text


def extract_from_json_tags(text: str) -> str:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        print(f"No content found within ```json---```. Returning oringial text.")
        return text
