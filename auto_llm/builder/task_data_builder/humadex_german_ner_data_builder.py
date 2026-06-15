import os
import string
from typing import List

from datasets import DatasetDict, Dataset, Features, Value, Sequence

import datasets
from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit

TAGS = {
    0: "O",
    1: "B-PROBLEM",
    2: "I-PROBLEM",
    3: "E-PROBLEM",
    4: "S-PROBLEM",
    5: "B-TREATMENT",
    6: "I-TREATMENT",
    7: "E-TREATMENT",
    8: "S-TREATMENT",
    9: "B-TEST",
    10: "I-TEST",
    11: "E-TEST",
    12: "S-TEST",
}


NER_FEATURES = Features(
    {
        TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None),
        TaskDatasetFeatures.OUTPUT_TEXT: {
            "PROBLEM": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "TREATMENT": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "TEST": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        },
    }
)


class HumadexGermanNerDataBuilder(TaskDataBuilder):
    """
    Data from https://huggingface.co/datasets/HUMADEX/german_ner_dataset
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:
        ds_dict = datasets.load_dataset("HUMADEX/german_ner_dataset")
        parsed_ds_dict = self.parse_data(ds_dict[DatasetSplit.TRAIN])

        return parsed_ds_dict

    def parse_data(self, ds):
        """
        Parse NER dataset by extracting entities from tag sequences.
        Groups consecutive tokens with the same entity type into entities.
        """
        input_texts = []
        all_ner_dicts = []

        for item in ds:
            text = " ".join(item["sentence"])
            ner_dicts = []

            # Extract entities by grouping consecutive tokens with the same entity type
            i = 0
            while i < len(item["tags"]):
                tag = item["tags"][i]

                if tag != 0:  # if not "O" (outside entity)
                    # Extract entity type (remove B-, I-, E-, S- prefixes)
                    tag_str = TAGS[tag]
                    entity_type = tag_str.replace("B-", "").replace("I-", "").replace("E-", "").replace("S-", "").strip()

                    # Collect all consecutive tokens belonging to this entity
                    entity_tokens = [item["sentence"][i]]
                    j = i + 1

                    while j < len(item["tags"]) and item["tags"][j] != 0:
                        next_tag_str = TAGS[item["tags"][j]]
                        next_entity_type = next_tag_str.replace("B-", "").replace("I-", "").replace("E-", "").replace("S-", "").strip()

                        if next_entity_type == entity_type:
                            entity_tokens.append(item["sentence"][j])
                            j += 1
                        else:
                            break

                    # Create NER dictionary for this entity
                    entity_text = " ".join(entity_tokens)
                    ner_dict = {"key": entity_type, "value": entity_text}
                    ner_dicts.append(ner_dict)

                    i = j
                else:
                    i += 1

            # Add to results if entities were found
            if ner_dicts:
                input_texts.append(text)
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
            sample_ner_dict = {key: [] for key in ner_keys}
            for item in ner_dict:
                sample_ner_dict[item["key"]] = [item["value"]]

            # skipping very short sentences
            if len(text.strip()) <= 10:
                continue

            count = 0
            for _, value in sample_ner_dict.items():
                if value != []:
                    count += 1

            # skipping samples with no labels
            if count == 0:
                continue

            samples[TaskDatasetFeatures.INPUT_TEXT].append(text)
            samples[TaskDatasetFeatures.OUTPUT_TEXT].append(sample_ner_dict)

        ds = Dataset.from_dict(samples, features=NER_FEATURES)
        ds_dict = self.split_ds(ds=ds)

        return ds_dict


if __name__ == "__main__":
    builder = HumadexGermanNerDataBuilder()
    ds_dict = builder.build()
    print(ds_dict)
    # print(ds_dict[ds_dict.keys()[0]][0])

    repo_id = "llm-4-kmu/humadex-german-ner"  # f"{hf_repo_id}/{dataset_name}"
    ds_dict.push_to_hub(
        repo_id=repo_id,
        token=os.getenv("HF_TOKEN"),
    )
