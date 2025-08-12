from aenum import StrEnum


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
