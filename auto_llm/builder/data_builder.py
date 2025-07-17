from abc import ABC, abstractmethod
from datasets import DatasetDict, load_from_disk

class DataBuilder(ABC):
    @abstractmethod
    def build(self) -> DatasetDict:
        ...
