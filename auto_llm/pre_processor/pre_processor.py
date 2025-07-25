from abc import abstractmethod, ABC
from typing import Dict, List

from transformers import BatchEncoding


class PreProcessor(ABC):
    @abstractmethod
    def pre_process(
        self,
        examples: Dict[str, List],
        max_length: int,
        truncation: bool,
    ) -> BatchEncoding: ...
