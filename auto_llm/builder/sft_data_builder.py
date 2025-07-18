import json
from typing import Dict, List

from datasets import DatasetDict, Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, AutoTokenizer
from transformers.utils import PaddingStrategy

from auto_llm.builder.data_builder import DataBuilder

LABEL_SEPERATOR_TOKEN = "<|label|>"
SEED = 0


def preprocess_fn(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    instruction_input_separator: str = "\n",
    input_output_separator: str = "\n",
    truncation: bool = True,
    padding: PaddingStrategy = PaddingStrategy.MAX_LENGTH,
    only_completion_loss: bool = True,
    apply_chat_template: bool = False,
    keep_instruction_with_system_content: bool = True,
) -> BatchEncoding:
    inputs_with_outputs = []
    inputs_only = []
    for instruction, input, expected_output in zip(
        examples["instruction"], examples["input"], examples["expected_output"]
    ):
        input_only = (
            instruction + instruction_input_separator + input + input_output_separator
        )
        input_with_output = input_only + expected_output + tokenizer.eos_token

        if apply_chat_template:
            if keep_instruction_with_system_content:
                conversation_with_input_only = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": input},
                ]
            else:
                conversation_with_input_only = [
                    {
                        "role": "user",
                        "content": instruction + instruction_input_separator + input,
                    },
                ]
            conversation_with_input_output = []
            conversation_with_input_output.extend(conversation_with_input_only)
            conversation_with_input_output.append(
                {"role": "assistant", "content": expected_output}
            )

            input_only = tokenizer.apply_chat_template(
                conversation=conversation_with_input_only,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_with_output = tokenizer.apply_chat_template(
                conversation=conversation_with_input_output,
                tokenize=False,
            )

        inputs_with_outputs.append(input_with_output)
        inputs_only.append(input_only)

    inputs_only_encodings = tokenizer(
        text=inputs_only,
        return_length=True,
        return_tensors="pt",
        add_special_tokens=not apply_chat_template,
    )

    inputs_with_outputs_encodings = tokenizer(
        text=inputs_with_outputs,
        return_tensors="pt",
        add_special_tokens=not apply_chat_template,
    )

    labels = inputs_with_outputs_encodings["input_ids"].clone()

    if only_completion_loss:
        for labels_list, length in zip(labels, inputs_only_encodings["length"]):
            labels_list[:length] = -100

    # truncation
    if truncation:
        truncate_length = max_length
    else:
        truncate_length = None

    encoded_inputs = {
        "input_ids": inputs_with_outputs_encodings["input_ids"][
            :, :truncate_length
        ].tolist(),
        "attention_mask": inputs_with_outputs_encodings["attention_mask"][
            :, :truncate_length
        ].tolist(),
        "labels": labels[:, :truncate_length].tolist(),
    }

    # tokenizer.pad does not pad keys other than input_ids. Hence, padding separately.
    encoded_inputs = {
        "input_ids": pad(
            encodings=encoded_inputs["input_ids"],
            max_length=max_length,
            padding_side=tokenizer.padding_side,
            pad_token_id=tokenizer.pad_token_id,
        ),
        "attention_mask": pad(
            encodings=encoded_inputs["attention_mask"],
            max_length=max_length,
            padding_side=tokenizer.padding_side,
            pad_token_id=0,
        ),
        # while padding labels, pad with -100 if only completion loss. Else use the default pad token.
        "labels": pad(
            encodings=encoded_inputs["labels"],
            max_length=max_length,
            padding_side=tokenizer.padding_side,
            pad_token_id=-100 if only_completion_loss else tokenizer.pad_token_id,
        ),
    }

    encodings = BatchEncoding(data=encoded_inputs)
    encodings = encodings.convert_to_tensors(tensor_type="pt")

    return encodings


def pad(
    encodings: List[List[int]], max_length: int, padding_side: str, pad_token_id: int
) -> List[List[int]]:
    padded_encodings = []
    for encoding in encodings:
        difference = max_length - len(encoding)

        if padding_side == "right":
            encoding = encoding + [pad_token_id] * difference
        else:
            encoding = [pad_token_id] * difference + encoding
        padded_encodings.append(encoding)
    return padded_encodings


class SftDataBuilder(DataBuilder):
    def __init__(self, dataset_path: str, tokenizer: PreTrainedTokenizer):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer

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
