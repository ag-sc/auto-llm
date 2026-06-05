import logging
from abc import ABC, abstractmethod

from datasets import DatasetDict, Dataset

from auto_llm.dto.builder_config import DatasetSplit

SEED = 0


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

    @staticmethod
    def split_ds(ds: Dataset) -> DatasetDict:
        ds_dict_sp_1 = ds.train_test_split(test_size=0.1, shuffle=True, seed=SEED)
        ds_dict_sp_2 = ds_dict_sp_1["test"].train_test_split(test_size=0.5, shuffle=True, seed=SEED)

        ds_dict = DatasetDict(
            {
                DatasetSplit.TRAIN.value: ds_dict_sp_1["train"],
                DatasetSplit.VALIDATION.value: ds_dict_sp_2["train"],
                DatasetSplit.TEST.value: ds_dict_sp_2["test"],
            }
        )

        return ds_dict
