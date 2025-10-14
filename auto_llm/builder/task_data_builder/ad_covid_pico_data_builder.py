import os

from datasets import DatasetDict, Dataset, Features, Value, Sequence
from collections import OrderedDict
from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit


class AdCovidPicoDataBuilder(TaskDataBuilder):
    """
    Data from https://github.com/BIDS-Xu-Lab/section_specific_annotation_of_PICO/tree/main/data
    Works both for AD and Covid-19 splits
    """

    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def build(self) -> DatasetDict:
        all_samples = {
            TaskDatasetFeatures.INPUT_TEXT: [],
            TaskDatasetFeatures.OUTPUT_TEXT: [],
        }

        for subdir, dirs, files in os.walk(self.raw_data_path):
            if not len(dirs):
                for file in files:
                    data_path = os.path.join(subdir, file)

                    # Check only files inside the "fold<x>" folders. Skip others.
                    if "fold" not in data_path:
                        continue
                    self.logger.info(f"Checking {data_path}")
                    samples = self.construct_pico_data(data_path=data_path)

                    for inp_text, out_text in zip(
                        samples[TaskDatasetFeatures.INPUT_TEXT],
                        samples[TaskDatasetFeatures.OUTPUT_TEXT],
                    ):
                        if inp_text in all_samples[TaskDatasetFeatures.INPUT_TEXT]:
                            continue

                        all_samples[TaskDatasetFeatures.INPUT_TEXT].append(inp_text)
                        all_samples[TaskDatasetFeatures.OUTPUT_TEXT].append(out_text)

        # features define the schema of the dataset. If not passed, the order of PICO keys would change.
        features = Features(
            {
                TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None),
                TaskDatasetFeatures.OUTPUT_TEXT: {
                    "P": Sequence(
                        feature=Value(dtype="string", id=None), length=-1, id=None
                    ),
                    "I": Sequence(
                        feature=Value(dtype="string", id=None), length=-1, id=None
                    ),
                    "C": Sequence(
                        feature=Value(dtype="string", id=None), length=-1, id=None
                    ),
                    "O": Sequence(
                        feature=Value(dtype="string", id=None), length=-1, id=None
                    ),
                },
            }
        )

        ds = Dataset.from_dict(all_samples, features=features)
        ds_dict = self.split_ds(ds=ds)
        return ds_dict

    @staticmethod
    def read_data_file(data_path: str):
        with open(data_path, "r") as f:
            data = f.readlines()

        return data

    def construct_pico_data(self, data_path: str):
        data = self.read_data_file(data_path)
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

                samples[TaskDatasetFeatures.INPUT_TEXT].append(inp_text)
                # samples["texts"].append(texts)
                # samples[TaskDatasetFeatures.OUTPUT_TEXT].append(entities)

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

                extracted_entities = OrderedDict({"P": [], "I": [], "C": [], "O": []})
                for form in entities_form:
                    entity_text = " ".join(texts[form["start_idx"] : form["stop_idx"]])

                    if entity_text not in extracted_entities[form["entity_key"]]:
                        extracted_entities[form["entity_key"]].append(entity_text)

                samples[TaskDatasetFeatures.OUTPUT_TEXT].append(extracted_entities)

                texts = []
                entities = []

            else:
                line = line.strip()
                if line:
                    sp = line.split("\t")
                    texts.append(sp[0])
                    try:
                        entities.append(sp[1])
                    except IndexError:
                        entities.append("-NA-")
        return samples
