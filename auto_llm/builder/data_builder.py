import json
import random
from enum import Enum
from typing import Dict, Any, List

from datasets import DatasetDict, Dataset, load_from_disk

SEED = 0


class SftDatasetType(str, Enum):
    CONVERSATIONAL = "conversational"
    PROMPT_COMPLETIONS = "prompt_completions"


# TODO: examples always look for {{examples}} tag. Assert if this is the case
#   need examples separator tag? Example {idx}: --> need example header?
#   sometime "validation" sometimes "dev".. make this common
#   instruction template or just str
class DataBuilder:
    def __init__(
        self,
        dataset_dir_or_path: str,
        instruction_template: str,
        input_template: str,
        output_template: str,
        dataset_type: SftDatasetType,
        test_size: float = 0.3,
        val_size: float = 0.1,
        instruction_input_separator: str = None,
        use_system_message: bool = None,
        parse_output_as_json: bool = False,
        num_few_shot_examples: int = None,
        few_shot_examples_split: str = None,
    ):
        """

        :param dataset_dir_or_path:
        :param instruction_template:
        :param input_template:
        :param output_template:
        :param dataset_type:
        :param test_size:
        :param val_size:
        :param instruction_input_separator:
        :param use_system_message:
        :param parse_output_as_json:
        """
        self.dataset_dir_or_path = dataset_dir_or_path
        self.test_size = test_size
        self.val_size = val_size

        self.instruction_template = instruction_template.strip()
        self.input_template = input_template.strip()
        self.output_template = output_template.strip()

        self.dataset_type = dataset_type
        self.instruction_input_separator = instruction_input_separator
        self.use_system_message = use_system_message

        self.parse_output_as_json = parse_output_as_json

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_examples_split = few_shot_examples_split

        if self.num_few_shot_examples >= 1:
            assert self.few_shot_examples_split is not None

        if self.dataset_type == SftDatasetType.CONVERSATIONAL:
            assert (
                self.use_system_message is not None
            ), f"Please set the parameter `use_system_message`. You have to specify whether to use System message if you want to build a {SftDatasetType.CONVERSATIONAL} dataset."
            if not self.use_system_message:
                assert (
                    self.instruction_input_separator is not None
                ), f"Please set the parameter `instruction_input_separator`. You have to specify how to separate the instruction and the input if you want to build a {SftDatasetType.CONVERSATIONAL} dataset without a system message."

        if self.dataset_type == SftDatasetType.PROMPT_COMPLETIONS:
            assert (
                self.instruction_input_separator is not None
            ), f"Please set the parameter `instruction_input_separator`. You have to specify how to separate the instruction and the input if you want to build a {SftDatasetType.PROMPT_COMPLETIONS} dataset."

    # def build(self) -> DatasetDict:
    #     # TODO: add support for other file formats
    #     # TODO: load dataset from HF Hub
    #
    #     dataset = load_dataset("json", data_files=self.dataset_path, split="all")
    #     dataset = self.split_dataset(
    #         dataset=dataset, test_size=self.test_size, val_size=self.val_size
    #     )
    #     return dataset

    def build(self) -> DatasetDict:
        ds_dict = load_from_disk(self.dataset_dir_or_path)
        ds_dict = ds_dict.map(
            function=self.construct_samples, batched=True, load_from_cache_file=False
        )

        if self.num_few_shot_examples >= 1:
            few_shot_split = ds_dict.get(self.few_shot_examples_split)
            ds_dict = ds_dict.map(
                function=self.add_few_shot_examples,
                batched=True,
                load_from_cache_file=False,
                fn_kwargs={"few_shot_split": few_shot_split},
            )

        return ds_dict

    def construct_samples(self, ds_items: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        if self.dataset_type == SftDatasetType.PROMPT_COMPLETIONS:
            samples = self._construct_samples_prompt_completions_ds(ds_items=ds_items)
        elif self.dataset_type == SftDatasetType.CONVERSATIONAL:
            samples = self._construct_samples_conversational_ds(ds_items=ds_items)
        return samples

    def add_few_shot_examples(
        self, ds_items: Dict[str, List[Any]], few_shot_split: Dataset
    ) -> Dict[str, List[Any]]:
        if self.dataset_type == SftDatasetType.PROMPT_COMPLETIONS:
            samples = self._add_few_shot_examples_prompt_completion_ds(
                ds_items=ds_items, few_shot_split=few_shot_split
            )
        elif self.dataset_type == SftDatasetType.CONVERSATIONAL:
            samples = self._add_few_shot_examples_conversational_ds(
                ds_items=ds_items, few_shot_split=few_shot_split
            )
        return samples

    def _add_few_shot_examples_prompt_completion_ds(
        self, ds_items: Dict[str, List[Any]], few_shot_split: Dataset
    ) -> Dict[str, List[Any]]:
        prompts = []
        for prompt in ds_items["prompt"]:
            examples = random.sample(
                few_shot_split["examples"], self.num_few_shot_examples
            )
            examples = [example.replace("{{examples}}", "") for example in examples]
            examples_str = ""
            for idx, example in enumerate(examples):
                examples_str += f"Example {idx+1}:\n{example}\n"
            prompts.append(prompt.replace("{{examples}}", examples_str))

        ds_items["prompt"] = prompts

        return ds_items

    def _add_few_shot_examples_conversational_ds(
        self, ds_items: Dict[str, List[Any]], few_shot_split: Dataset
    ) -> Dict[str, List[Any]]:
        new_messages = []
        for messages in ds_items["messages"]:
            examples = random.sample(
                few_shot_split["examples"], self.num_few_shot_examples
            )
            examples = [example.replace("{{examples}}", "") for example in examples]
            examples_str = ""
            for idx, example in enumerate(examples):
                examples_str += f"Example {idx+1}:\n{example}\n"

            _new_messages = []
            for message in messages:
                if message["role"] == "user":
                    message["content"] = message["content"].replace(
                        "{{examples}}", examples_str
                    )
                _new_messages.append(message)
            new_messages.append(_new_messages)

        ds_items["messages"] = new_messages

        return ds_items

    def _construct_samples_prompt_completions_ds(
        self, ds_items: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        samples = {"prompt": [], "completions": [], "examples": []}

        for text, entities in zip(ds_items["text"], ds_items["entities"]):
            # instruction_text = self.get_instruction_text(text=text)
            instruction_text = self.instruction_template
            input_text = self.get_input_text(text=text)
            prompt = self.get_prompt(
                instruction_text=instruction_text, input_text=input_text
            )
            completion = self.get_completion(entities=entities)
            example = input_text + "\n" + completion

            samples["prompt"].append(prompt)
            samples["completions"].append(completion)
            samples["examples"].append(example)

        return samples

    def _construct_samples_conversational_ds(
        self, ds_items: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        samples = {"messages": [], "examples": []}

        for text, entities in zip(ds_items["text"], ds_items["entities"]):
            instruction_text = self.instruction_template
            input_text = self.get_input_text(text=text)
            prompt = self.get_prompt(
                instruction_text=instruction_text, input_text=input_text
            )
            completion = self.get_completion(entities=entities)
            example = input_text + "\n" + completion

            if self.use_system_message:
                messages = [
                    {"role": "system", "content": instruction_text},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": completion},
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]

            samples["messages"].append(messages)
            samples["examples"].append(example)

        return samples

    # TODO: key "text" and "entities" are task-specific
    def get_instruction_text(self, text: str) -> str:
        return self.instruction_template.format(text=text.strip())

    def get_input_text(self, text: str) -> str:
        return self.input_template.format(text=text.strip(), examples="{{examples}}")

    def get_prompt(self, instruction_text: str, input_text: str) -> str:
        return instruction_text + self.instruction_input_separator + input_text

    def get_completion(self, entities: str) -> str:
        if self.parse_output_as_json:
            entities = json.dumps(entities, indent=4).strip()
        completion = self.output_template.format(entities=entities)
        return completion

    @staticmethod
    def split_dataset(
        dataset: Dataset,
        test_size: float,
        val_size: float,
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
