from datasets import DatasetDict, load_dataset, Dataset

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import (
    DatasetSplit,
    TaskDatasetFeatures,
)


class MedmcqaDataBuilder(TaskDataBuilder):
    """Build the MedMCQA dataset into ``input_text`` / ``output_text`` format.

    Source: ``openlifescienceai/medmcqa`` on HuggingFace.

    Raw columns: ``question``, ``opa``/``opb``/``opc``/``opd`` (four options),
    ``cop`` (0-indexed correct option: 0=A, 1=B, 2=C, 3=D), ``choice_type``
    (``"single"`` or ``"multi"``).

    """

    OPTION_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

    def build(self) -> DatasetDict:
        ds_dict = load_dataset("openlifescienceai/medmcqa", trust_remote_code=True)

        return DatasetDict(
            {
                DatasetSplit.TRAIN: self._process_split(ds_dict["train"]),
                DatasetSplit.VALIDATION: self._process_split(ds_dict["validation"]),
            }
        )

    def _process_split(self, ds: Dataset) -> Dataset:
        samples = []
        for item in ds:
            question = item["question"]
            input_text = (
                f"Question: {question}\n"
                f"Options:\n"
                f"A. {item['opa']}\n"
                f"B. {item['opb']}\n"
                f"C. {item['opc']}\n"
                f"D. {item['opd']}"
            )
            output_text = self.OPTION_MAP[item["cop"]]

            samples.append(
                {
                    TaskDatasetFeatures.INPUT_TEXT: input_text,
                    TaskDatasetFeatures.OUTPUT_TEXT: output_text,
                }
            )

        return Dataset.from_list(samples)
