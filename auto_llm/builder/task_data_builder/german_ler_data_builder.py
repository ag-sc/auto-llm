import os
import string
from typing import List

from datasets import DatasetDict, Dataset, Features, Value, Sequence

import datasets
from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit

# features define the schema of the dataset. If not passed, the order of keys would change.
NER_FEATURES = Features(
    {
        TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None),
        TaskDatasetFeatures.OUTPUT_TEXT: {
            "PER": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "RR": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "AN": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "LD": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "ST": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "STR": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "ORG": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "UN": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "INN": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "GRT": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "MRK": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "GS": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "VO": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "EUN": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "VS": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "VT": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        },
    }
)

COARSE_NER_FEATURES = Features(
    {
        TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None),
        TaskDatasetFeatures.OUTPUT_TEXT: {
            "PER": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "LOC": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "ORG": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "NRM": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "REG": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "RS": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        },
    }
)


class GermanLerNerDataBuilder(TaskDataBuilder):
    """
    Data from https://huggingface.co/datasets/elenanereiss/german-ler
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:
        ds_dict = datasets.load_dataset("elenanereiss/german-ler")

        parsed_ds_dict = DatasetDict(
            {
                DatasetSplit.TRAIN.value: self.parse_data(ds_dict["train"]),
                DatasetSplit.VALIDATION.value: self.parse_data(ds_dict["validation"]),
                DatasetSplit.TEST.value: self.parse_data(ds_dict["test"]),
            }
        )
        return parsed_ds_dict

    def parse_data(self, ds, ner_key: str = "ner", features: Features = NER_FEATURES):
        input_texts = []
        all_ner_dicts = []
        for item in ds:
            tokens = item["tokens"]
            ner = item[ner_key]

            ner_dicts = []
            for idx, ner_label in enumerate(ner):
                if "B-" in ner_label:
                    try:
                        o_idx = ner[idx:].index("O")
                    except:
                        print("No O tag found, Skipping!")
                        continue
                    next_o_idx = idx + o_idx
                    ner_value = tokens[idx:next_o_idx]
                    ner_value = self._join_tokens(tokens=ner_value)
                    ner_label = ner_label.replace("B-", "").strip()
                    ner_dict = dict(key=ner_label, value=ner_value)
                    ner_dicts.append(ner_dict)

            if len(ner_dicts) != 0:
                tokens_concat = self._join_tokens(tokens=tokens)
                input_texts.append(tokens_concat)
                all_ner_dicts.append(ner_dicts)

        ner_keys = []
        for ner_dict in all_ner_dicts:
            for item in ner_dict:
                if item["key"] not in ner_keys:
                    ner_keys.append(item["key"])
        samples = {
            TaskDatasetFeatures.INPUT_TEXT: [],
            TaskDatasetFeatures.OUTPUT_TEXT: [],
        }
        for text, ner_dict in zip(input_texts, all_ner_dicts):
            sample_ner_dict = {key: "" for key in ner_keys}
            for item in ner_dict:
                sample_ner_dict[item["key"]] = [item["value"]]

            samples[TaskDatasetFeatures.INPUT_TEXT].append(text)
            samples[TaskDatasetFeatures.OUTPUT_TEXT].append(sample_ner_dict)

        ds = Dataset.from_dict(samples, features=features)
        print(ds)
        print(ds[0])

        return ds

    def _join_tokens(self, tokens: List[str]) -> str:
        """
        Joins tokens, ensuring punctuation does not have leading spaces.
        """
        punctuation = set(string.punctuation)
        result = []

        for i, token in enumerate(tokens):
            # Add a space if it's not the first token AND it's not punctuation
            if i > 0 and token not in punctuation:
                result.append(" ")
            result.append(token)

        return "".join(result)


if __name__ == "__main__":
    builder = GermanLerNerDataBuilder()
    ds_dict = builder.build()

    repo_id = "llm-4-kmu/german-ler-ner"  # f"{hf_repo_id}/{dataset_name}"
    ds_dict.push_to_hub(
        repo_id=repo_id,
        token=os.getenv("HF_TOKEN"),
    )
