import json
import random
from typing import Dict, Any, List

from datasets import DatasetDict, Dataset, load_from_disk

from auto_llm.builder.trainer_data_builder.trainer_data_builder import (
    TrainerDataBuilder,
)
from auto_llm.dto.builder_config import (
    SftDatasetType,
    PromptCompletionDatasetFeatures,
    TaskDatasetFeatures,
    ConversationalDatasetFeatures,
)

SEED = 0


# TODO: examples always look for {{examples}} tag. Assert if this is the case
#   need examples separator tag? Example {idx}: --> need example header?
#   instruction template or just str
class SftDataBuilder(TrainerDataBuilder):
    def __init__(
        self,
        dataset_dir: str,
        instruction_template: str,
        input_template: str,
        output_template: str,
        dataset_type: SftDatasetType,
        instruction_input_separator: str = None,
        use_system_message: bool = None,
        parse_output_as_json: bool = False,
        num_few_shot_examples: int = None,
        few_shot_examples_split: str = None,
    ):
        """

        :param dataset_dir:
        :param instruction_template:
        :param input_template:
        :param output_template:
        :param instruction_input_separator:
        :param use_system_message:
        :param parse_output_as_json:
        :param num_few_shot_examples:
        :param few_shot_examples_split:
        """
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type

        self.instruction_template = instruction_template.strip()
        self.input_template = input_template.strip()
        self.output_template = output_template.strip()

        self.instruction_input_separator = instruction_input_separator
        self.use_system_message = use_system_message

        self.parse_output_as_json = parse_output_as_json

        self.num_few_shot_examples = num_few_shot_examples
        self.few_shot_examples_split = few_shot_examples_split

        self.sanity_check()

    def build(self) -> DatasetDict:
        # TODO: refactor. there could be many other options to consider
        #   local or remote with all the above options
        ds_dict = load_from_disk(self.dataset_dir)
        ds_dict = ds_dict.map(
            function=self.construct_samples,
            batched=True,
            load_from_cache_file=False,
        )

        if self.num_few_shot_examples and self.num_few_shot_examples >= 1:
            few_shot_split = ds_dict.get(self.few_shot_examples_split)
            ds_dict = ds_dict.map(
                function=self.add_few_shot_examples,
                batched=True,
                load_from_cache_file=False,
                fn_kwargs={"few_shot_split": few_shot_split},
            )

        return ds_dict

    def sanity_check(self):
        raise NotImplementedError

    def construct_samples(self, ds_items: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        raise NotImplementedError

    def add_few_shot_examples(
        self, ds_items: Dict[str, List[Any]], few_shot_split: Dataset
    ) -> Dict[str, List[Any]]:
        raise NotImplementedError

    def get_instruction_text(self, text: str) -> str:
        return self.instruction_template.format(
            input_text=text.strip()
        )  # input_text = TaskDatasetFeatures.INPUT_TEXT

    def get_input_text(self, text: str) -> str:
        return self.input_template.format(
            input_text=text.strip(), examples="{{examples}}"
        )  # input_text = TaskDatasetFeatures.INPUT_TEXT, examples = PromptCompletionDatasetFeatures.EXAMPLES or ConversationalDatasetFeatures.EXAMPLES

    def get_prompt(self, instruction_text: str, input_text: str) -> str:
        return instruction_text + self.instruction_input_separator + input_text

    def get_completion(self, entities: str) -> str:
        if self.parse_output_as_json:
            entities = json.dumps(entities, indent=4).strip()
        completion = self.output_template.format(
            output_text=entities
        )  # output_text = TaskDatasetFeatures.OUTPUT_TEXT
        return completion


class PromptCompletionsSftDataBuilder(SftDataBuilder):
    def sanity_check(self):
        if self.num_few_shot_examples and self.num_few_shot_examples >= 1:
            assert self.few_shot_examples_split is not None

        assert (
            self.instruction_input_separator is not None
        ), f"Please set the parameter `instruction_input_separator`. You have to specify how to separate the instruction and the input if you want to build a {SftDatasetType.PROMPT_COMPLETIONS} dataset."

    def construct_samples(self, ds_items: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        samples = {
            PromptCompletionDatasetFeatures.PROMPT: [],
            PromptCompletionDatasetFeatures.COMPLETION: [],
            PromptCompletionDatasetFeatures.EXAMPLES: [],
        }

        for text, entities in zip(
            ds_items[TaskDatasetFeatures.INPUT_TEXT],
            ds_items[TaskDatasetFeatures.OUTPUT_TEXT],
        ):
            # instruction_text = self.get_instruction_text(text=text)
            instruction_text = self.instruction_template
            input_text = self.get_input_text(text=text)
            prompt = self.get_prompt(
                instruction_text=instruction_text, input_text=input_text
            )
            completion = self.get_completion(entities=entities)
            example = input_text + "\n" + completion

            samples[PromptCompletionDatasetFeatures.PROMPT].append(prompt)
            samples[PromptCompletionDatasetFeatures.COMPLETION].append(completion)
            samples[PromptCompletionDatasetFeatures.EXAMPLES].append(example)

        return samples

    def add_few_shot_examples(
        self, ds_items: Dict[str, List[Any]], few_shot_split: Dataset
    ) -> Dict[str, List[Any]]:
        prompts = []
        for prompt in ds_items[PromptCompletionDatasetFeatures.PROMPT]:
            examples = random.sample(
                few_shot_split[PromptCompletionDatasetFeatures.EXAMPLES],
                self.num_few_shot_examples,
            )
            examples = [example.replace("{{examples}}", "") for example in examples]
            examples_str = ""
            for idx, example in enumerate(examples):
                examples_str += f"Example {idx+1}:\n{example}\n"
            prompts.append(prompt.replace("{{examples}}", examples_str))

        ds_items[PromptCompletionDatasetFeatures.PROMPT] = prompts

        return ds_items


class ConversationalSftDataBuilder(SftDataBuilder):
    def sanity_check(self):
        if self.num_few_shot_examples and self.num_few_shot_examples >= 1:
            assert self.few_shot_examples_split is not None

        assert (
            self.use_system_message is not None
        ), f"Please set the parameter `use_system_message`. You have to specify whether to use System message if you want to build a {SftDatasetType.CONVERSATIONAL} dataset."
        if not self.use_system_message:
            assert (
                self.instruction_input_separator is not None
            ), f"Please set the parameter `instruction_input_separator`. You have to specify how to separate the instruction and the input if you want to build a {SftDatasetType.CONVERSATIONAL} dataset without a system message."

    def construct_samples(self, ds_items: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        samples = {
            ConversationalDatasetFeatures.MESSAGE: [],
            ConversationalDatasetFeatures.EXAMPLES: [],
        }

        for text, entities in zip(
            ds_items[TaskDatasetFeatures.INPUT_TEXT],
            ds_items[TaskDatasetFeatures.OUTPUT_TEXT],
        ):
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

            samples[ConversationalDatasetFeatures.MESSAGE].append(messages)
            samples[ConversationalDatasetFeatures.EXAMPLES].append(example)

        return samples

    def add_few_shot_examples(
        self, ds_items: Dict[str, List[Any]], few_shot_split: Dataset
    ) -> Dict[str, List[Any]]:
        new_messages = []
        for messages in ds_items[ConversationalDatasetFeatures.MESSAGE]:
            examples = random.sample(
                few_shot_split[ConversationalDatasetFeatures.EXAMPLES],
                self.num_few_shot_examples,
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

        ds_items[ConversationalDatasetFeatures.MESSAGE] = new_messages

        return ds_items
