from aenum import StrEnum
from pydantic import BaseModel, Field


class DatasetSplit(StrEnum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class TaskDatasetFeatures(StrEnum):
    INPUT_TEXT = "input_text"
    OUTPUT_TEXT = "output_text"


class PromptCompletionDatasetFeatures(StrEnum):
    PROMPT = "prompt"
    COMPLETION = "completion"
    EXAMPLES = "examples"


class ConversationalDatasetFeatures(StrEnum):
    MESSAGE = "message"
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
    dataset_type: SftDatasetType
    instruction_input_separator: str = None
    use_system_message: bool = None
    parse_output_as_json: bool = False
    num_few_shot_examples: int = None
    few_shot_examples_split: str = None
