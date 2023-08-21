import abc
from abc import ABC
from dataclasses import dataclass

class State(ABC):
    @abc.abstractmethod
    def elements(self):
        pass

    @abc.abstractmethod
    def toSeq(self):
        pass

    @abc.abstractmethod
    def isEmpty(self):
        pass

class EmptyState(State):
    def elements(self):
        return 0

    def toSeq(self):
        return []

    def isEmpty(self):
        return True