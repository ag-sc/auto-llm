import os

from datasets import DatasetDict, Dataset, Features, Value, Sequence

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures
from collections import OrderedDict

keys = [
    "PERS",
    "LOC",
    "ORG",
    "CAMP",
    "DATE",
    "GHETTO",
]

feature_keys = {key: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None) for key in keys}

# features define the schema of the dataset. If not passed, the order of keys would change.
NER_FEATURES = Features({TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None), TaskDatasetFeatures.OUTPUT_TEXT: feature_keys})


class EhriHolocaustNerDataBuilder(TaskDataBuilder):
    """
    Data from https://github.com/EHRI/EHRI-NER/tree/main
    Used this split: https://github.com/EHRI/EHRI-NER/blob/main/dataset/iob/de/ehri_de.txt

    """

    def __init__(self): ...

    def build(self) -> DatasetDict:
        with open("/vol/auto_llm/raw_datasets/ehri_de.txt", "r") as f:
            data = f.readlines()

        doc_start = "-DOCSTART-\n"
        new_data = []
        for x in data:
            new_data.append(x)

            if x == "\n":
                new_data.append(doc_start)

        samples = self.parse_data(data=new_data)
        ds = Dataset.from_dict(samples, features=NER_FEATURES)
        ds_dict = self.split_ds(ds=ds)
        print(ds_dict)

        return ds_dict

    def parse_data(self, data):
        samples = {
            TaskDatasetFeatures.INPUT_TEXT: [],
            TaskDatasetFeatures.OUTPUT_TEXT: [],
        }
        texts = []
        entities = []
        for line in data:
            if "-DOCSTART-" in line:
                inp_text = " ".join(texts)

                if not len(texts) > 1:
                    continue

                if inp_text in samples[TaskDatasetFeatures.INPUT_TEXT]:
                    continue

                entities_form = []
                for idx, entity in enumerate(entities):
                    if "B-" in entity:
                        entity_key = entity.split("-")[-1]
                        start_idx = idx
                        stop_idx = idx
                        for next_idx in range(idx + 1, len(entities)):
                            if entities[next_idx] == "O":
                                stop_idx = next_idx
                                break
                        idx = stop_idx
                        entities_form.append(
                            {
                                "entity_key": entity_key,
                                "start_idx": start_idx,
                                "stop_idx": stop_idx,
                            }
                        )

                extracted_entities = OrderedDict({key: [] for key in keys})
                for form in entities_form:
                    entity_text = " ".join(texts[form["start_idx"] : form["stop_idx"]])

                    if entity_text not in extracted_entities[form["entity_key"]]:
                        extracted_entities[form["entity_key"]].append(entity_text)

                count = 0
                for _, value in extracted_entities.items():
                    if value != []:
                        count += 1

                # skipping very short sentences
                if len(inp_text) <= 10:
                    continue

                # skipping samples with no labels
                if count != 0:
                    samples[TaskDatasetFeatures.INPUT_TEXT].append(inp_text)
                    samples[TaskDatasetFeatures.OUTPUT_TEXT].append(extracted_entities)

                texts = []
                entities = []

            else:
                line = line.strip()
                if line:
                    sp = line.split(" ")
                    texts.append(sp[0])
                    try:
                        entities.append(sp[1])
                    except IndexError:
                        entities.append("-NA-")
        return samples


if __name__ == "__main__":
    builder = EhriHolocaustNerDataBuilder()
    ds_dict = builder.build()
    print(ds_dict)

    # repo_id = "llm-4-kmu/ehri-holocaust-ner"  # f"{hf_repo_id}/{dataset_name}"
    # ds_dict.push_to_hub(
    #     repo_id=repo_id,
    #     token=os.getenv("HF_TOKEN"),
    # )
