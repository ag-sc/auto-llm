import json
from typing import Dict, List

from datasets import DatasetDict, Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, AutoTokenizer

from auto_llm.builder.data_builder import DataBuilder

LABEL_SEPERATOR_TOKEN = "<|label|>"
SEED = 0


class SftDataBuilder(DataBuilder):
    def __init__(self, dataset_path: str, tokenizer: PreTrainedTokenizer):
        self.dataset_path = dataset_path

        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(LABEL_SEPERATOR_TOKEN, special_tokens=True)
        self.label_separator_token_id = self.tokenizer.convert_tokens_to_ids(
            LABEL_SEPERATOR_TOKEN
        )
        # model.resize_token_embeddings(len(tokenizer))

    def load_dataset(self) -> Dataset:
        # TODO: add support for other file formats
        # TODO: load dataset from HF Hub
        with open(self.dataset_path, "r") as f:
            data = json.load(f)

        dataset = Dataset.from_list(data)
        return dataset

    @staticmethod
    def split_dataset(
        dataset: Dataset,
        test_size: float = 0.3,
        val_size: float = 0.1,
    ) -> DatasetDict:
        # get train and the test+val splits
        ds_dict_sp_one = dataset.train_test_split(test_size=test_size, seed=SEED)

        # split test+val splits to get test and val splits
        ds_dict_sp_two = ds_dict_sp_one["test"].train_test_split(
            test_size=val_size, seed=SEED
        )

        ds_dict = DatasetDict(
            {
                "train": ds_dict_sp_one["train"],
                "test": ds_dict_sp_two["train"],
                "val": ds_dict_sp_two["test"],
            }
        )

        return ds_dict

    def tokenize(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = dataset_dict.map(
            function=self.preprocess_fn,
            batched=True,
            # fn_kwargs={
            #     "max_context_length": max_context_length,
            #     "instruction_input_separator": instruction_input_separator,
            # },
        )

        return dataset_dict

    def preprocess_fn(
        self,
        examples: Dict[str, List[str]],
        max_length: int = 512,
        instruction_input_separator: str = "\n",
    ) -> BatchEncoding:
        inputs_with_outputs = []
        inputs_only = []
        for instruction, input, expected_output in zip(
            examples["instruction"], examples["input"], examples["expected_output"]
        ):
            input_only = instruction + instruction_input_separator + input
            input_with_output = input_only + expected_output

            inputs_with_outputs.append(input_with_output)
            inputs_only.append(input_only)

        inputs_only_encodings = self.tokenizer(
            text=inputs_only, return_length=True, return_tensors="pt"
        )

        inputs_with_outputs_encodings = self.tokenizer(
            text=inputs_with_outputs, return_tensors="pt"
        )

        labels = inputs_with_outputs_encodings["input_ids"].clone()

        for labels_list, length in zip(labels, inputs_only_encodings["length"]):
            labels_list[:length] = -100

        # truncation
        encoded_inputs = {
            "input_ids": inputs_with_outputs_encodings["input_ids"][:, :max_length],
            "attention_mask": inputs_with_outputs_encodings["attention_mask"][
                :, :max_length
            ],
            "labels": labels[:, :max_length],
        }

        encoded_inputs = self.tokenizer.pad(
            encoded_inputs=encoded_inputs, padding="max_length", max_length=max_length
        )

        return encoded_inputs

    def build(self) -> DatasetDict: ...


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/gemma-2-2b",
        # padding_side=...
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset_path = "data/train.json"

    builder = SftDataBuilder(dataset_path=dataset_path, tokenizer=tokenizer)
    ds = builder.load_dataset()
    ds_dict = builder.split_dataset(ds)

    encodings = builder.tokenize(dataset_dict=ds_dict)
    print(encodings)
    ...
