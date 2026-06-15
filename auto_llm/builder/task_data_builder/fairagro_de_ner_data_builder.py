import os

import datasets
from datasets import DatasetDict, Dataset, Features, Value, Sequence

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit

keys = [
    "city", "country", "cropSpecies", "cropVariety", "duration", "endTime",
    "region", "soilDepth", "soilOrganicCarbon", "soilPH", "soilReferenceGroup",
    "soilTexture", "startTime",
]

NER_FEATURES = Features({
    TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None),
    TaskDatasetFeatures.OUTPUT_TEXT: {
        key: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None)
        for key in keys
    },
})


class FairagroDeNerDataBuilder(TaskDataBuilder):
    """
    Data from https://huggingface.co/datasets/IT-ZBMED/Agriculture_NER_Dataset_for_FAIR_Metadata_Enrichment
    Uses the sentence_split config, filtered to German texts only (Language == "de").
    No local file download required.
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:
        raw = datasets.load_dataset(
            "IT-ZBMED/Agriculture_NER_Dataset_for_FAIR_Metadata_Enrichment",
            "sentence_split",
        )

        # Keep only German sentences
        raw = DatasetDict({
            split: raw[split].filter(lambda x: x["Language"] == "de")
            for split in raw
        })

        # Carve 10% validation from train; use original test set
        train_val = raw["train"].train_test_split(test_size=0.1, seed=42)

        ds_dict = DatasetDict({
            DatasetSplit.TRAIN.value:      self._convert(train_val["train"]),
            DatasetSplit.VALIDATION.value: self._convert(train_val["test"]),
            DatasetSplit.TEST.value:       self._convert(raw["test"]),
        })
        print(ds_dict)
        return ds_dict

    def _convert(self, hf_split) -> Dataset:
        rows = []
        for row in hf_split:
            spans = {key: [] for key in keys}
            tokens = row["Tokens"]
            bio_labels = row["Labels"]
            i = 0
            while i < len(bio_labels):
                lbl = bio_labels[i]
                if lbl.startswith("B-"):
                    base = lbl[2:]
                    j = i + 1
                    while j < len(bio_labels) and bio_labels[j] == f"I-{base}":
                        j += 1
                    span_text = " ".join(tokens[i:j])
                    if span_text not in spans[base]:
                        spans[base].append(span_text)
                    i = j
                else:
                    i += 1
            rows.append({
                TaskDatasetFeatures.INPUT_TEXT: " ".join(tokens),
                TaskDatasetFeatures.OUTPUT_TEXT: spans,
            })
        return Dataset.from_list(rows, features=NER_FEATURES)


if __name__ == "__main__":
    builder = FairagroDeNerDataBuilder()
    ds_dict = builder.build()

    repo_id = "llm-4-kmu/FAIRagroDE"
    ds_dict.push_to_hub(
        repo_id=repo_id,
        token=os.getenv("HF_TOKEN"),
    )
