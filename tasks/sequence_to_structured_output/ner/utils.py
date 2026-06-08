import ast
import json
import re
from typing import Any, Dict, List, Union
from thefuzz import fuzz


def clean_and_extract_json(text: str) -> str:
    def extract_from_tags(text, pattern):
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    # Pattern 1: Perfectly closed markdown block
    strict_pattern = r"```json\s*(.*?)\s*```"
    match = extract_from_tags(text=text, pattern=strict_pattern)

    # Pattern 2 Fallback: Look for open-ended json block if strict failed
    if not match:
        # The '$' ensures it grabs everything up to the very end of the string
        fallback_pattern = r"```json\s*(.*?)$"
        match = extract_from_tags(text=text, pattern=fallback_pattern)

    # Return the matched string, or the raw text if no markdown block was found
    return match if match else text.strip()


def parse_dict(text: str) -> Union[Dict, None]:
    """Safely parses text into a dictionary, accommodating trailing commas."""
    if not text:
        return None
    try:
        # Attempt robust parsing (handles trailing commas common in LLM outputs)
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            # Fallback to strict JSON
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse text as a dict/JSON: {text[:100]}...")
            return None


def process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, float]:
    """Function to compute the metrics given the input and the generated response."""
    print("Response:\n", result)

    expected_entities_dict = doc.get("output_text", {})
    predicted_response = result[0] if result else ""

    print("[DEBUG] expected_entities_dict", expected_entities_dict)
    print("[DEBUG] predicted_response", predicted_response)

    extracted_text = clean_and_extract_json(text=predicted_response)
    predicted_response_dict = parse_dict(text=extracted_text)

    print("[DEBUG] predicted_response_dict", predicted_response_dict)

    exact_match_score = 0.0
    partial_match_score = 0.0
    fuzzy_match_score = 0.0
    f1_score = 0.0

    if not isinstance(predicted_response_dict, dict):
        print("Cannot parse response, cannot compute score. Keeping scores 0")
        print("----------------------------")
        return {
            "exact_match": 0.0,
            "partial_match": 0.0,
            "fuzzy_match": 0.0,
            "f1_score": 0.0,
        }

    num_entity_keys_with_values = 0
    for key, expected_value in expected_entities_dict.items():
        if not isinstance(expected_value, list):
            expected_value = [str(expected_value)] if expected_value else []

        num_entity_keys_with_values += 1

        # FIX: Safe default is an empty list, not "NA string"
        predicted_value = predicted_response_dict.get(key, [])
        if not isinstance(predicted_value, list):
            predicted_value = [str(predicted_value)] if predicted_value else []

        print(f"\nKey: {key}, Expected: {expected_value}, Predicted: {predicted_value}")

        # 1. Exact match between the Lists
        full_match = set(expected_value) == set(predicted_value)
        exact_match_score += float(full_match)

        # 2. Partial match (Exact token containment)
        partial_match = 0
        for item in expected_value:
            if item in predicted_value:
                partial_match += 1
        try:
            partial_match /= len(expected_value)
        except ZeroDivisionError:
            partial_match = 0.0
        partial_match_score += partial_match

        # 3. Fuzzy ratio match
        fuzzy_match = 0
        for exp_item in expected_value:
            all_ratios = []
            for pred_item in predicted_value:
                all_ratios.append(fuzz.ratio(str(exp_item), str(pred_item)) / 100)
            fuzzy_match += max(all_ratios, default=0.0)
        try:
            fuzzy_match /= len(expected_value)
        except ZeroDivisionError:
            fuzzy_match = 0.0
        fuzzy_match_score += fuzzy_match

        # 4. F1 Score
        exp_set = set(expected_value)
        pred_set = set(predicted_value)

        tp = len(exp_set & pred_set)
        fp = len(pred_set - exp_set)
        fn = len(exp_set - pred_set)

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

    # Guard against completely empty configurations
    if num_entity_keys_with_values > 0:
        exact_match_score /= num_entity_keys_with_values
        partial_match_score /= num_entity_keys_with_values
        fuzzy_match_score /= num_entity_keys_with_values
        f1_score /= num_entity_keys_with_values

    print("Final exact_match_score:", exact_match_score)
    print("Final partial_match_score:", partial_match_score)
    print("Final fuzzy_match_score:", fuzzy_match_score)
    print("Final f1_score:", f1_score)
    print("----------------------------")

    return {
        "exact_match": exact_match_score,
        "partial_match": partial_match_score,
        "fuzzy_match": fuzzy_match_score,
        "f1_score": f1_score,
    }
