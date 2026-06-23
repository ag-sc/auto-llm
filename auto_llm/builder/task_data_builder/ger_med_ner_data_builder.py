import json
import os

from datasets import DatasetDict, Dataset, Features, Value, Sequence

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures


class GerMedNerDataBuilder(TaskDataBuilder):
    """
    Data from https://github.com/frankkramer-lab/GERNERMED/tree/main
    Save the file GERNERMED_dataset.json from https://github.com/frankkramer-lab/GERNERMED/blob/main/data/GERNERMED_dataset.json
    in the path ``/vol/auto_llm/raw_datasets`` before running the builder.
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:

        with open("/vol/auto_llm/raw_datasets/GERNERMED_dataset.json", "r") as f:
            data = json.load(f)

        keys = []
        for x in data:
            for item in x["annotations"]:
                k = item["type"]
                if k not in keys:
                    keys.append(k)
        print("keys", keys)

        feature_keys = {key: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None) for key in keys}

        # features define the schema of the dataset. If not passed, the order of keys would change.
        NER_FEATURES = Features({TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None), TaskDatasetFeatures.OUTPUT_TEXT: feature_keys})

        samples = {
            TaskDatasetFeatures.INPUT_TEXT: [],
            TaskDatasetFeatures.OUTPUT_TEXT: [],
        }
        long_count = 0
        for x in data:
            ner_dict = {key: [] for key in keys}
            de_sent = x["de"]
            for item in x["annotations"]:
                de_entity_name = item["type"]
                de_span = item["de_spans"]
                de_entity_value = de_sent[de_span[0] : de_span[1]]

                if de_entity_value not in ner_dict[de_entity_name]:
                    ner_dict[de_entity_name].append(de_entity_value)

                de_text_len = len(de_entity_value)
                en_text_len = len(item["content"])

                # TODO: there are many occurences where the DE translation is quite longer than the EN text.
                # For now, we are skipping such examples.
                if de_text_len - en_text_len > 10:
                    long_count += 1
                    # print("WARNING: de_text_len is very long!!")
                    # print("de_text:", de_text)
                    # print("en_text:", item["content"])
                    # print()
                    continue

                # skipping very short sentences
                if len(de_sent) <= 10:
                    continue

                # skipping samples with no labels
                count = 0
                for _, value in ner_dict.items():
                    if value != []:
                        count += 1
                if count == 0:
                    continue

                samples[TaskDatasetFeatures.INPUT_TEXT].append(de_sent)
                samples[TaskDatasetFeatures.OUTPUT_TEXT].append(ner_dict)

        print("long_count", long_count)
        print("data len", len(data))

        print("len samples", len(samples[TaskDatasetFeatures.INPUT_TEXT]))

        ds = Dataset.from_dict(samples, features=NER_FEATURES)
        ds_dict = self.split_ds(ds=ds)

        return ds_dict


if __name__ == "__main__":
    builder = GerMedNerDataBuilder()
    ds_dict = builder.build()
    print(ds_dict)

    # repo_id = "llm-4-kmu/ger-med-ner"  # f"{hf_repo_id}/{dataset_name}"
    # ds_dict.push_to_hub(
    #     repo_id=repo_id,
    #     token=os.getenv("HF_TOKEN"),
    # )
