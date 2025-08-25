from abc import ABC, abstractmethod

from datasets import DatasetDict


class TrainerDataBuilder(ABC):
    """
    Construct trainer-specific data builder. This class accepts DatasetDicts from TaskDataBuilder and builds a `DatasetDict`.
    - DatasetDict has keys :py:class:`auto_llm.builder.utils.DatasetSplit`
    - Each Dataset (for example: DatasetDict["train"]) has keys specific to the trainer.
        - Prompt Completions SFT Trainer should have keys from PromptCompletionDatasetFeatures
        - Conversational SFT Trainer should have keys from ConversationDatasetFeatures
    """

    @abstractmethod
    def build(self) -> DatasetDict: ...
