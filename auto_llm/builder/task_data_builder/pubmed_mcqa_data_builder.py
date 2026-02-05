from datasets import DatasetDict, load_dataset, Dataset

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures


class PubMedMcqaDataBuilder(TaskDataBuilder):
    def build(self) -> DatasetDict:
        ds_dict = load_dataset(
            "qiaojin/PubMedQA", name="pqa_labeled", trust_remote_code=True
        )
        ds = ds_dict["train"]

        samples = []
        for item in ds:
            question = item["question"]
            context = "\n".join(item["context"]["contexts"])
            input_text = f"Abstract: {context}\nQuestion: {question}"
            output_text = item["final_decision"]

            sample = {
                TaskDatasetFeatures.INPUT_TEXT: input_text,
                TaskDatasetFeatures.OUTPUT_TEXT: output_text,
            }

            samples.append(sample)

        ds = Dataset.from_list(samples)
        ds_dict = self.split_ds(ds=ds)

        return ds_dict
