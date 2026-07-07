import ast
import json
import re
from typing import Any, Dict, List, Union
from thefuzz import fuzz

# Corpus-level (micro) F1 config. partial_f1 counts a gold/pred entity pair as a
# match when their token_set_ratio is at or above the threshold (lenient match).
MICRO_FUZZY_SCORER = fuzz.token_set_ratio
MICRO_FUZZY_THRESHOLD = 0.90

# TODO: Is this the right approach? When there is no reference,
# f1-score is usually undefined. But this reduces the overall score.
# How to deal with this?
# For now, awarding a score of 1.0 for all metrics


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


def get_f1_score(expected_value, predicted_value):
    if expected_value == [] and predicted_value == []:
        return 1.0

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
    return f1


def get_fuzzy_match(expected_value, predicted_value):
    if expected_value == [] and predicted_value == []:
        return 1.0

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
    return fuzzy_match


def get_partial_match(expected_value, predicted_value):
    if expected_value == [] and predicted_value == []:
        return 1.0

    partial_match = 0
    for item in expected_value:
        if item in predicted_value:
            partial_match += 1
    try:
        partial_match /= len(expected_value)
    except ZeroDivisionError:
        partial_match = 0.0
    return partial_match


def get_exact_match(expected_value, predicted_value):
    if expected_value == [] and predicted_value == []:
        return 1.0

    exact_match_score = set(expected_value) == set(predicted_value)
    return float(exact_match_score)


def _exact_counts(gold_list, pred_list):
    gs, ps = set(gold_list), set(pred_list)
    tp = len(gs & ps)
    return tp, len(ps - gs), len(gs - ps)  # tp, fp, fn


def _fuzzy_counts(gold_list, pred_list):
    # One-to-one greedy matching above the threshold; each entity used once.
    pairs = []
    for gi, g in enumerate(gold_list):
        for pi, p in enumerate(pred_list):
            s = MICRO_FUZZY_SCORER(g, p) / 100.0
            if s >= MICRO_FUZZY_THRESHOLD:
                pairs.append((s, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p = set(), set()
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
    tp = len(used_g)
    return tp, len(pred_list) - tp, len(gold_list) - tp  # tp, fp, fn


def get_micro_counts(expected_entities_dict: Dict[str, List], predicted_response_dict: Dict[str, List]):
    """Pool (tp, fp, fn) over all labels of one sample for exact and fuzzy matching.

    Returned per-sample and summed across the corpus by the micro_f1 / partial_f1
    aggregations, so the reported F1 is a true corpus-level micro score.
    """
    if not isinstance(predicted_response_dict, dict):
        predicted_response_dict = {}

    e_tp = e_fp = e_fn = 0
    f_tp = f_fp = f_fn = 0
    for label in set(expected_entities_dict) | set(predicted_response_dict):
        g = _to_str_list(expected_entities_dict.get(label, []))
        p = _to_str_list(predicted_response_dict.get(label, []))
        tp, fp, fn = _exact_counts(g, p)
        e_tp += tp; e_fp += fp; e_fn += fn
        tp, fp, fn = _fuzzy_counts(g, p)
        f_tp += tp; f_fp += fp; f_fn += fn
    return (e_tp, e_fp, e_fn), (f_tp, f_fp, f_fn)


def _to_str_list(value):
    if not isinstance(value, list):
        value = [value] if value else []
    return [str(v).strip() for v in value if str(v).strip() != ""]


def _micro_f1_from_counts(items):
    tp = sum(i[0] for i in items)
    fp = sum(i[1] for i in items)
    fn = sum(i[2] for i in items)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def micro_f1(items):
    """Aggregation: exact-string micro F1 over pooled per-sample (tp, fp, fn)."""
    return _micro_f1_from_counts(items)


def partial_f1(items):
    """Aggregation: fuzzy (token_set_ratio) micro F1 over pooled per-sample counts."""
    return _micro_f1_from_counts(items)


def process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, float]:
    """Function to compute the metrics given the input and the generated response."""
    expected_entities_dict = doc.get("output_text", {})

    predicted_response = result[0] if result else ""
    extracted_text = clean_and_extract_json(text=predicted_response)
    predicted_response_dict = parse_dict(text=extracted_text)

    """
    `per_label_scores`: Dict[str, Dict[str, float]] --> Metric values for each label per sample
        {
            "label-name": 
                {
                    "metric-1-name": metric-1-value,
                    "metric-2-name": metric-2-value,
                    ...
                }
        }

    `per_sample_scores`: Dict[str, float] --> Metric values for each sample
        {
            "metric-1-name": metric-1-value,
            "metric-2-name": metric-2-value,
            ...
        }
    """

    per_label_scores: Dict[str, Dict[str, float]] = get_per_label_scores(expected_entities_dict=expected_entities_dict, predicted_response_dict=predicted_response_dict)
    per_sample_scores: Dict[str, float] = get_per_sample_scores(expected_entities_dict=expected_entities_dict, predicted_response_dict=predicted_response_dict)

    consolidated_metrics_dict = {}
    for key, value in per_label_scores.items():
        for metric_name, metric_value in value.items():
            consolidated_metrics_dict[f"{metric_name}_{key}"] = metric_value

    for metric_name, metric_value in per_sample_scores.items():
        consolidated_metrics_dict[f"{metric_name}_all_labels"] = metric_value

    # Emit per-sample counts for the corpus-level (micro) F1 aggregations.
    exact_counts, fuzzy_counts = get_micro_counts(expected_entities_dict, predicted_response_dict)
    consolidated_metrics_dict["micro_f1"] = exact_counts
    consolidated_metrics_dict["partial_f1"] = fuzzy_counts

    print("Input:", doc.get("input_text", {}))
    print("Expected Output:", expected_entities_dict)
    print("Generated Output:", predicted_response_dict)
    print("Consolidated Metrics:", json.dumps(consolidated_metrics_dict, indent=4))
    print("-----\n\n")

    return consolidated_metrics_dict


def get_per_sample_scores(expected_entities_dict: Dict[str, List], predicted_response_dict: Dict[str, List]):
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
        expected_value = [str(v) for v in expected_value]

        num_entity_keys_with_values += 1

        predicted_value = predicted_response_dict.get(key, [])
        if not isinstance(predicted_value, list):
            predicted_value = [str(predicted_value)] if predicted_value else []
        predicted_value = [str(v) for v in predicted_value]

        # 1. Exact match between the Lists
        exact_match = get_exact_match(expected_value, predicted_value)
        exact_match_score += exact_match

        # 2. Partial match (Exact token containment)
        partial_match = get_partial_match(expected_value, predicted_value)
        partial_match_score += partial_match

        # 3. Fuzzy ratio match
        fuzzy_match = get_fuzzy_match(expected_value, predicted_value)
        fuzzy_match_score += fuzzy_match

        # 4. F1 Score
        f1 = get_f1_score(expected_value, predicted_value)
        f1_score += f1

    if num_entity_keys_with_values > 0:
        exact_match_score /= num_entity_keys_with_values
        partial_match_score /= num_entity_keys_with_values
        fuzzy_match_score /= num_entity_keys_with_values
        f1_score /= num_entity_keys_with_values

    return {
        "exact_match": exact_match_score,
        "partial_match": partial_match_score,
        "fuzzy_match": fuzzy_match_score,
        "f1_score": f1_score,
    }


def get_per_label_scores(expected_entities_dict: Dict[str, List], predicted_response_dict: Dict[str, List]) -> Dict[str, Dict[str, float]]:
    per_label_scores_dict = {label: {"exact_match": 0.0, "partial_match": 0.0, "fuzzy_match": 0.0, "f1_score": 0.0} for label in expected_entities_dict.keys()}

    if not isinstance(predicted_response_dict, dict):
        print("Cannot parse response, cannot compute score. Keeping scores 0")
        print("----------------------------")
        return per_label_scores_dict

    for key, expected_value in expected_entities_dict.items():
        if not isinstance(expected_value, list):
            expected_value = [str(expected_value)] if expected_value else []
        expected_value = [str(v) for v in expected_value]

        predicted_value = predicted_response_dict.get(key, [])
        if not isinstance(predicted_value, list):
            predicted_value = [str(predicted_value)] if predicted_value else []
        predicted_value = [str(v) for v in predicted_value]

        # 1. Exact match between the Lists
        exact_match = get_exact_match(expected_value, predicted_value)
        per_label_scores_dict[key]["exact_match"] = exact_match

        # 2. Partial match (Exact token containment)
        partial_match = get_partial_match(expected_value, predicted_value)
        per_label_scores_dict[key]["partial_match"] = partial_match

        # 3. Fuzzy ratio match
        fuzzy_match = get_fuzzy_match(expected_value, predicted_value)
        per_label_scores_dict[key]["fuzzy_match"] = fuzzy_match

        # 4. F1 Score
        f1_score = get_f1_score(expected_value, predicted_value)
        per_label_scores_dict[key]["f1_score"] = f1_score

    return per_label_scores_dict
