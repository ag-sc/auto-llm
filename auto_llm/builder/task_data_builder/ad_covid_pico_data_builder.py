import os

from datasets import DatasetDict, Dataset

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.builder.utils import TaskDatasetFeatures, DatasetSplit


class AdCovidPicoDataBuilder(TaskDataBuilder):
    """
    Data from https://github.com/BIDS-Xu-Lab/section_specific_annotation_of_PICO/tree/main/data
    Works both for AD and Covid-19 splits
    """

    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def build(self) -> DatasetDict:
        train_samples = {
            TaskDatasetFeatures.INPUT_TEXT: [],
            TaskDatasetFeatures.OUTPUT_TEXT: [],
        }
        dev_samples = {
            TaskDatasetFeatures.INPUT_TEXT: [],
            TaskDatasetFeatures.OUTPUT_TEXT: [],
        }
        test_samples = {
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
                    if "train" in file:
                        train_samples[TaskDatasetFeatures.INPUT_TEXT].extend(
                            samples[TaskDatasetFeatures.INPUT_TEXT]
                        )
                        train_samples[TaskDatasetFeatures.OUTPUT_TEXT].extend(
                            samples[TaskDatasetFeatures.OUTPUT_TEXT]
                        )
                    elif "dev" in file:
                        dev_samples[TaskDatasetFeatures.INPUT_TEXT].extend(
                            samples[TaskDatasetFeatures.INPUT_TEXT]
                        )
                        dev_samples[TaskDatasetFeatures.OUTPUT_TEXT].extend(
                            samples[TaskDatasetFeatures.OUTPUT_TEXT]
                        )
                    elif "test" in file:
                        test_samples[TaskDatasetFeatures.INPUT_TEXT].extend(
                            samples[TaskDatasetFeatures.INPUT_TEXT]
                        )
                        test_samples[TaskDatasetFeatures.OUTPUT_TEXT].extend(
                            samples[TaskDatasetFeatures.OUTPUT_TEXT]
                        )

        train_ds = Dataset.from_dict(train_samples)
        dev_ds = Dataset.from_dict(dev_samples)
        test_ds = Dataset.from_dict(test_samples)

        ds_dict = DatasetDict(
            {
                DatasetSplit.TRAIN: train_ds,
                DatasetSplit.VALIDATION: dev_ds,
                DatasetSplit.TEST: test_ds,
            }
        )
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
                if not len(texts) > 1:
                    continue
                samples[TaskDatasetFeatures.INPUT_TEXT].append(" ".join(texts))
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

                extracted_entities = {"P": [], "I": [], "C": [], "O": []}
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
