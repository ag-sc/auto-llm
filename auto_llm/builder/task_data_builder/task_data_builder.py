import logging
from abc import ABC, abstractmethod

from datasets import DatasetDict


class TaskDataBuilder(ABC):
    """
    Construct task-specific data builder. This class accepts data from different sources and builds a `DatasetDict`.
    - DatasetDict has keys :py:class:`auto_llm.builder.utils.DatasetSplit` - train, test and validation. All keys should be present
    - Each Dataset (for example: DatasetDict["train"]) has keys :py:class:`auto_llm.builder.utils.TaskDatasetFeatures`
    """

    @abstractmethod
    def build(self) -> DatasetDict: ...

    @property
    def logger(self):
        return logging.getLogger(name=self.__class__.__name__)

    def save(self, ds_dict: DatasetDict, path: str):
        ds_dict.save_to_disk(dataset_dict_path=path)
        self.logger.info(f"Saved to {path}")
