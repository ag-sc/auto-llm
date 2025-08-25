from abc import abstractmethod, ABC

from accelerate import Accelerator
from accelerate.logging import get_logger

from auto_llm.dto.trainer_run_config import TrainerRunConfig

accelerator = Accelerator()


class TrainerWrapper(ABC):
    def __init__(self, config: TrainerRunConfig):
        self.config = config

    @property
    def logger(self):
        return get_logger(name=self.__class__.__name__)

    @abstractmethod
    def run(self): ...
