from abc import abstractmethod, ABC

from accelerate import Accelerator
from accelerate.logging import get_logger

accelerator = Accelerator()


class TrainerWrapper(ABC):
    @abstractmethod
    def run(self): ...

    @property
    def logger(self):
        return get_logger(name=self.__class__.__name__)
