from abc import ABC, abstractmethod


class Estimator(ABC):
    @abstractmethod
    def estimate(self): ...
