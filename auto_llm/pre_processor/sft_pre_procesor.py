import copy
from typing import Dict, List, Tuple

from datasets import DatasetDict
from transformers import BatchEncoding, PreTrainedTokenizer

from auto_llm.pre_processor.pre_processor import PreProcessor

Message = List[Dict[str, str]]


class SftPreProcessor(PreProcessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        completion_only_loss: bool,
        prompt_completion_separator: str = "\n",
    ):
        self.tokenizer = tokenizer
        self.completion_only_loss = completion_only_loss
        self.prompt_completion_separator = prompt_completion_separator

    def pre_process(
        self,
        examples: Dict[str, List],
        max_length: int,
        truncation: bool,
    ) -> BatchEncoding:
        is_conversational = self.is_conversational(examples)

        self.check_dataset_tokenizer_compatibility(is_conversational=is_conversational)

        input_sequences, full_sequences = self.collect_sequences(
            examples=examples, is_conversational=is_conversational
        )

        input_sequences_encodings = self.tokenizer(
            text=input_sequences,
            return_length=True,
            add_special_tokens=not is_conversational,
        )

        full_sequences_encodings = self.tokenizer(
            text=full_sequences,
            add_special_tokens=not is_conversational,
        )

        labels = self.collect_labels(
            full_sequences_encodings=full_sequences_encodings,
            input_sequences_encodings=input_sequences_encodings,
        )

        encodings = {**full_sequences_encodings, "labels": labels}

        if truncation:
            encodings = self.truncate(encodings=encodings, max_length=max_length)

        encodings = {
            "input_ids": self.pad(
                encodings=encodings["input_ids"],
                max_length=max_length,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.tokenizer.pad_token_id,
            ),
            "attention_mask": self.pad(
                encodings=encodings["attention_mask"],
                max_length=max_length,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=0,
            ),
            # while padding labels, pad with -100 if only completion loss. Else use the default pad token.
            "labels": self.pad(
                encodings=encodings["labels"],
                max_length=max_length,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=(
                    -100 if self.completion_only_loss else self.tokenizer.pad_token_id
                ),
            ),
        }

        encodings = BatchEncoding(data=encodings)
        encodings = encodings.convert_to_tensors(tensor_type="pt")

        return encodings

    @staticmethod
    def is_conversational(examples: Dict[str, List[str]]) -> bool:
        if "prompt" in examples.keys() and "completion" in examples.keys():
            return False
        elif "messages" in examples.keys():
            return True
        else:
            raise Exception(
                "The passed examples are neither conversational or completion!"
            )

    @staticmethod
    def is_dataset_conversational(dataset_dict: DatasetDict) -> bool:
        ds = dataset_dict["train"]
        if "prompt" in ds.column_names and "completion" in ds.column_names:
            return False
        elif "messages" in ds.column_names:
            return True
        else:
            raise Exception(
                "The passed examples are neither conversational or completion!"
            )

    def collect_sequences(self, examples: Dict[str, List], is_conversational: bool):
        if is_conversational:
            return self.collect_sequences_conversational(examples=examples)
        else:
            return self.collect_sequences_completions(examples=examples)

    def collect_sequences_completions(
        self, examples: Dict[str, List]
    ) -> Tuple[List[str], List[str]]:
        input_sequences = []
        full_sequences = []

        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            input_sequence = prompt + self.prompt_completion_separator
            full_sequence = input_sequence + completion + self.tokenizer.eos_token

            input_sequences.append(input_sequence)
            full_sequences.append(full_sequence)

        return input_sequences, full_sequences

    def collect_sequences_conversational(
        self, examples: Dict[str, List]
    ) -> Tuple[List[str], List[str]]:
        input_messages: List[Message] = []
        full_messages: List[Message] = []
        for messages in examples["messages"]:
            for idx, message in enumerate(messages):
                if message["role"] == "assistant":
                    last_assistant_idx = idx
                    break
            else:
                raise Exception("Assistant turn not found!")

            input_messages.append(messages[:last_assistant_idx])

            full_message = messages
            # full_message[last_assistant_idx]["content"] += self.tokenizer.eos_token
            full_messages.append(full_message)

        input_sequences = self.apply_chat_template_to_messages(
            messages=input_messages, tokenize=False, add_generation_prompt=True
        )

        full_sequences = self.apply_chat_template_to_messages(
            messages=full_messages, tokenize=False, add_generation_prompt=False
        )

        return input_sequences, full_sequences

    def collect_labels(self, full_sequences_encodings, input_sequences_encodings):
        labels = copy.deepcopy(full_sequences_encodings["input_ids"])
        if self.completion_only_loss:
            for labels_list, length in zip(labels, input_sequences_encodings["length"]):
                labels_list[:length] = [-100] * length

        return labels

    def apply_chat_template_to_messages(
        self, messages: List[Message], tokenize: bool, add_generation_prompt: bool
    ) -> List[str]:
        sequences = []
        for message in messages:
            sequences.append(
                self.tokenizer.apply_chat_template(
                    conversation=message,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        return sequences

    @staticmethod
    def truncate(
        encodings: Dict[str, List[List[int]]], max_length: int
    ) -> Dict[str, List[List[int]]]:
        truncated_encodings = {}
        for key, value in encodings.items():
            items = []
            for item in value:
                item = item[:max_length]
                items.append(item)
            truncated_encodings[key] = items

        return truncated_encodings

    @staticmethod
    def pad(
        encodings: List[List[int]],
        max_length: int,
        padding_side: str,
        pad_token_id: int,
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

    def check_dataset_tokenizer_compatibility(self, is_conversational: bool):
        # A. Conversational DS         + Tokenizer w/ Chat Template    --> Match
        # B. Conversational DS         + Tokenizer w/o Chat Template   --> Exception - tokenizing system/user tokens differently
        # C. Non-Conversational DS     + Tokenizer w/ Chat Template    --> Warning
        # D. Non-Conversational DS     + Tokenizer w/o Chat Template   --> Match
        # TODO: mostly non-instruct models do not contain a chat template. However, this is not always the case. Any better way to check if or if not chat model?

        if self.tokenizer.chat_template:
            if not is_conversational:
                # case C.
                print(
                    f"[WARNING] You are using a non-conversational dataset, but with a tokenizer with chat template. "
                    f"Better usages will be: (i) Conversational DS + Tokenizer w/ Chat Template or (ii) "
                    f"Non-Conversational DS + Tokenizer w/o Chat Template."
                )
        else:
            if is_conversational:
                # case B.
                raise Exception(
                    f"You are using a conversational dataset, but with a tokenizer without chat template. Please use a tokenizer with a chat template."
                )
