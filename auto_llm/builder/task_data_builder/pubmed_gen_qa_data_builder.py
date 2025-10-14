from datasets import DatasetDict, load_dataset, Dataset

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit


class PubMedGenQaDataBuilder(TaskDataBuilder):
    def build(self) -> DatasetDict:
        ds_dict = load_dataset(
            "qiaojin/PubMedQA", name="pqa_labeled", trust_remote_code=True
        )
        ds = ds_dict["train"]

        samples = []
        for item in ds:
            if not item["final_decision"] == "yes":
                continue

            question = item["question"]
            context = "\n".join(item["context"]["contexts"])
            input_text = f"Abstract: {context}\nQuestion: {question}"
            output_text = item["long_answer"]

            sample = {
                TaskDatasetFeatures.INPUT_TEXT: input_text,
                TaskDatasetFeatures.OUTPUT_TEXT: output_text,
            }

            samples.append(sample)

        ds = Dataset.from_list(samples)
        ds_dict = self.split_ds(ds=ds)
        return ds_dict
