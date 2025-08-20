import json
import re
from typing import Dict, Any, List, Union
from thefuzz import fuzz


def process_results(doc: Dict[str, Any], result: List[str]):
    """
    Function to compute the metrics given the input and the generated response.

    :param doc: this includes the input
    :param result: this is the generated response
    :return: dict of metric key-value pairs

    TODO: constraint LLM output to JSONs. Use other libraries? https://github.com/1rgs/jsonformer

    TODO: define evaluation metrics.
     (1) exact match on list level
     (2) partial match on list level - exact match on content level
     (3) partial/fuzzy match on content level

    """

    print("Response:\n", result)

    expected_entities_dict = doc["output_text"]
    predicted_response = result[0]

    predicted_response = extract_from_tags(
        text=predicted_response, pattern=r"```json\s*(.*?)\s*```"
    )
    predicted_response = extract_from_tags(
        text=predicted_response, pattern=r"```json\s*(.*?)\s*"
    )

    predicted_response_dict = parse_dict(text=predicted_response)

    print("expected_entities_dict\n", expected_entities_dict)
    print("predicted_response_dict\n", predicted_response_dict)

    exact_match_score = 0
    partial_match_score = 0
    fuzzy_match_score = 0
    f1_score = 0

    if not isinstance(predicted_response_dict, dict):
        print("Cannot parse response, cannot compute score. Keeping scores 0")
        print("----------------------------")
        return {
            "exact_match": exact_match_score,
            "partial_match": partial_match_score,
            "fuzzy_match": fuzzy_match_score,
            "f1_score": f1_score,
        }

    num_entity_keys_with_values = 0
    for key, expected_value in expected_entities_dict.items():
        if len(expected_value) < 1:
            continue

        num_entity_keys_with_values += 1

        # if predicted response does not have the expected key, yield "NA".
        predicted_value = predicted_response_dict.get(key, "NA")
        print(f"\nKey: {key}, Expected: {expected_value}, Predicted: {predicted_value}")
        # expected and predicted value -> List[str]

        # exact match
        # checking for exact match between the Lists
        full_match = set(expected_value) == set(predicted_value)
        exact_match_score += full_match

        # partial match
        # checking for partial match between the Lists, but exact match between the entities
        partial_match = 0
        for item in expected_value:
            if item in predicted_value:
                partial_match += 1
        partial_match /= len(expected_value)
        partial_match_score += partial_match

        # fuzzy ratio
        # checking for fuzzy match between the entities
        fuzzy_match = 0
        for exp_item in expected_value:
            all_ratios = []
            for pred_item in predicted_value:
                all_ratios.append(fuzz.ratio(exp_item, pred_item) / 100)
            fuzzy_match += max(all_ratios, default=0)
        fuzzy_match /= len(expected_value)
        fuzzy_match_score += fuzzy_match

        # f1 score
        tp = len(set(expected_value) & set(predicted_value))
        fp = len(set(predicted_value) - set(expected_value))
        fn = len(set(expected_value) - set(predicted_value))
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        f1_score += f1

        print(f"Exact Match: {full_match}")
        print(f"Partial Match: {partial_match}")
        print(f"Fuzzy Match: {fuzzy_match}")
        print(f"F1 Score: {f1}")
        print("---------")

    exact_match_score /= num_entity_keys_with_values
    partial_match_score /= num_entity_keys_with_values
    fuzzy_match_score /= num_entity_keys_with_values
    f1_score /= num_entity_keys_with_values

    print("exact_match_score", exact_match_score)
    print("partial_match_score", partial_match_score)
    print("fuzzy_match_score", fuzzy_match_score)
    print("f1_score", f1_score)
    print("----------------------------")

    return {
        "exact_match": exact_match_score,
        "partial_match": partial_match_score,
        "fuzzy_match": fuzzy_match_score,
        "f1_score": f1_score,
    }


def parse_dict(text: str) -> Union[Dict | str]:
    try:
        data = json.loads(text)
        return data
    except:
        print(f"Can't parse JSON: {text}")
        return text


def extract_from_tags(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        print(f"No content found within {pattern}. Returning original text.")
        return text
