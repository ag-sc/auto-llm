from aenum import StrEnum
from pydantic import BaseModel


class DatasetSplit(StrEnum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class TaskDatasetFeatures(StrEnum):
    INPUT_TEXT = "input_text"
    OUTPUT_TEXT = "output_text"


class PromptCompletionDatasetFeatures(StrEnum):
    PROMPT = "prompt"
    COMPLETIONS = "completions"
    EXAMPLES = "examples"


class ConversationalDatasetFeatures(StrEnum):
    MESSAGE = "message"
    EXAMPLES = "examples"


class SftDatasetType(StrEnum):
    CONVERSATIONAL = "conversational"
    PROMPT_COMPLETIONS = "prompt_completions"


class TrainerDataBuilderConfig(BaseModel):
    dataset_dir: str
    instruction_template: str
    input_template: str
    output_template: str
    dataset_type: SftDatasetType
    instruction_input_separator: str = None
    use_system_message: bool = None
    parse_output_as_json: bool = False
    num_few_shot_examples: int = None
    few_shot_examples_split: str = None
