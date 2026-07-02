from datasets import DatasetDict, load_dataset, Dataset

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import (
    DatasetSplit,
    TaskDatasetFeatures,
)


class MedQaDataBuilder(TaskDataBuilder):

    def build(self) -> DatasetDict:
        ds_dict = load_dataset(
            "bigbio/med_qa",
            name="med_qa_en_source",
            trust_remote_code=True,
        )

        return DatasetDict(
            {
                DatasetSplit.TRAIN: self._process_split(ds_dict["train"]),
                DatasetSplit.VALIDATION: self._process_split(ds_dict["validation"]),
                DatasetSplit.TEST: self._process_split(ds_dict["test"]),
            }
        )

    def _process_split(self, ds: Dataset) -> Dataset:
        samples = []

        for item in ds:
            question = item["question"]

            options_text = "\n".join(
                [f"{opt['key']}. {opt['value']}" for opt in item["options"]]
            )

            input_text = (
                f"Question: {question}\n"
                f"Options:\n{options_text}"
            )

            output_text = item["answer_idx"]  # già tipo "A", "B", ...

            samples.append(
                {
                    TaskDatasetFeatures.INPUT_TEXT: input_text,
                    TaskDatasetFeatures.OUTPUT_TEXT: output_text,
                }
            )

        return Dataset.from_list(samples)