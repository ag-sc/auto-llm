from typing import Optional, Union

from aenum import StrEnum
from pydantic import BaseModel, Field


class DatasetSplit(StrEnum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class TaskDatasetFeatures(StrEnum):
    INPUT_TEXT = "input_text"
    OUTPUT_TEXT = "output_text"


class PromptPlaceholders(StrEnum):
    INPUT_TEXT = "{{input}}"
    OUTPUT_TEXT = "{{output}}"
    EXAMPLES_TEXT = "{{examples}}"


class PromptCompletionDatasetFeatures(StrEnum):
    PROMPT = "prompt"
    COMPLETION = "completion"
    EXAMPLES = "examples"


class ConversationalDatasetFeatures(StrEnum):
    MESSAGES = "messages"
    EXAMPLES = "examples"


class SftDatasetType(StrEnum):
    CONVERSATIONAL = "conversational"
    PROMPT_COMPLETIONS = "prompt_completions"


class TrainerDataBuilderConfig(BaseModel):
    dataset_dir: str = Field(
        title="Dataset Directory",
        description="The path where the dataset dictionary lies.",
    )
    instruction_template: str = Field(
        title="Instruction Template",
        description="The template for the instruction. Note keywords to be replaced should be enclosed within {{}} tags.",
    )
    input_template: str = Field(
        title="Input Template",
        description="The template for the input text. This is the text that goes to the 'user' field in case of an instruction tuned model. Note keywords to be replaced should be enclosed within {{}} tags.",
    )
    output_template: str = Field(
        title="Output Template",
        description="The template for the output text. This is the template for the expected response from the model. Note keywords to be replaced should be enclosed within {{}} tags.",
    )
    dataset_type: Union[SftDatasetType, str] = Field(
        title="Dataset Type",
        description="The type of dataset to be built which determines the specific formatting rules.",
    )
    instruction_input_separator: Optional[str] = Field(
        title="Instruction Input Separator",
        description="An optional string used to separate the instruction and input fields in the final formatted text.",
        default=None,
    )
    use_system_message: Optional[bool] = Field(
        title="Use System Message",
        description="An optional boolean flag to indicate whether a system message should be used while applying chat template. Only valid for Conversational Datasets",
        default=None,
    )
    parse_output_as_json: bool = Field(
        title="Parse Output as JSON",
        description="A boolean flag indicating if the output should be parsed as a JSON object, which is useful for tasks requiring structured outputs.",
        default=False,
    )
    num_few_shot_examples: Optional[int] = Field(
        title="Number of Few-Shot Examples",
        description="The number of few-shot examples to include in the formatted prompt for in-context learning.",
        default=None,
    )
    few_shot_examples_split: Optional[str] = Field(
        title="Few-Shot Examples Split",
        description="The name of the dataset split (e.g., 'train', 'validation') from which to select few-shot examples.",
        default=None,
    )
