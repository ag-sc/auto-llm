import pytest

from auto_llm.builder.trainer_data_builder.sft_data_builder import (
    PromptCompletionsSftDataBuilder,
    ConversationalSftDataBuilder,
)
from auto_llm.dto.builder_config import (
    TrainerDataBuilderConfig,
    DatasetSplit,
    PromptCompletionDatasetFeatures,
    SftDatasetType,
    ConversationalDatasetFeatures,
)


@pytest.fixture
def prompt_completions_data_builder_config() -> TrainerDataBuilderConfig:
    return TrainerDataBuilderConfig(
        dataset_dir="/vol/auto_llm/processed_datasets/pico/AD",
        instruction_template='Given the text "Text", extract the PICO tags in the JSON format "Format". Do not modify the sentences.\nFormat:\n```json\n{\n  "P": ["value for P"],\n  "I": ["value for I"],\n  "C": ["value for C"],\n  "O": ["value for O"],\n}\n```\n',
        input_template="Text: {{input}}\n\nPICO tags according to the format:\n",
        output_template="```json\n{{output}}\n```",
        dataset_type=SftDatasetType.PROMPT_COMPLETIONS,
        instruction_input_separator="\n",
        parse_output_as_json=True,
    )


@pytest.fixture
def conversational_data_builder_config() -> TrainerDataBuilderConfig:
    return TrainerDataBuilderConfig(
        dataset_dir="/vol/auto_llm/processed_datasets/pico/AD",
        instruction_template='Given the text "Text", extract the PICO tags in the JSON format "Format". Do not modify the sentences.\nFormat:\n```json\n{\n  "P": ["value for P"],\n  "I": ["value for I"],\n  "C": ["value for C"],\n  "O": ["value for O"],\n}\n```\n',
        input_template="Text: {{input}}\n\nPICO tags according to the format:\n",
        output_template="```json\n{{output}}\n```",
        dataset_type=SftDatasetType.CONVERSATIONAL,
        instruction_input_separator="\n",
        parse_output_as_json=True,
        use_system_message=False,
    )


def test_prompt_completions_data_builder(prompt_completions_data_builder_config):
    builder = PromptCompletionsSftDataBuilder(
        **prompt_completions_data_builder_config.model_dump()
    )
    ds_dict = builder.build()

    for split in iter(DatasetSplit):
        assert PromptCompletionDatasetFeatures.PROMPT in ds_dict[split].column_names
        assert PromptCompletionDatasetFeatures.COMPLETION in ds_dict[split].column_names


def test_conversational_data_builder(conversational_data_builder_config):
    builder = ConversationalSftDataBuilder(
        **conversational_data_builder_config.model_dump()
    )
    ds_dict = builder.build()

    for split in iter(DatasetSplit):
        assert ConversationalDatasetFeatures.MESSAGES in ds_dict[split].column_names
        assert ConversationalDatasetFeatures.EXAMPLES in ds_dict[split].column_names


def test_trainer_data_builder_with_remote_ds():
    trainer_data_builder_config = TrainerDataBuilderConfig(
        dataset_dir="llm-4-kmu/qa_pubmedqa",
        instruction_template="Answer the following question based on the given context.",
        input_template="{{input}}\nAnswer:",
        output_template="{{output}}",
        dataset_type=SftDatasetType.PROMPT_COMPLETIONS,
        instruction_input_separator="\n",
        parse_output_as_json=False,
        use_system_message=False,
        # num_few_shot_examples=1,
    )

    builder = PromptCompletionsSftDataBuilder(
        **trainer_data_builder_config.model_dump()
    )
    ds_dict = builder.build()

    for split in iter(DatasetSplit):
        assert PromptCompletionDatasetFeatures.PROMPT in ds_dict[split].column_names
        assert PromptCompletionDatasetFeatures.COMPLETION in ds_dict[split].column_names


def test_few_shot_examples():
    trainer_data_builder_config = TrainerDataBuilderConfig(
        dataset_dir="llm-4-kmu/qa_pubmedqa",
        instruction_template="Answer the following question based on the given context.",
        input_template="{{examples}}\n{{input}}\nAnswer:",
        output_template="{{output}}",
        dataset_type=SftDatasetType.PROMPT_COMPLETIONS,
        instruction_input_separator="\n",
        parse_output_as_json=False,
        use_system_message=False,
        num_few_shot_examples=5,
        few_shot_examples_split="validation",
    )

    builder = PromptCompletionsSftDataBuilder(
        **trainer_data_builder_config.model_dump()
    )
    ds_dict = builder.build()

    for split in iter(DatasetSplit):
        assert PromptCompletionDatasetFeatures.PROMPT in ds_dict[split].column_names
        assert PromptCompletionDatasetFeatures.COMPLETION in ds_dict[split].column_names
