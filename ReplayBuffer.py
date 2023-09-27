from dataclasses import dataclass
from typing import Any, List
import random
from abc import ABC, abstractmethod


@dataclass
class Experience:
    actualState: Any
    action: Any
    nextState: Any
    reward: float = 0.0


class ReplayBuffer(ABC):

    @abstractmethod
    def insert(self, actualState: Any, action: Any, reward: float, nextState: Any) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def subsample(self, batchSize: int) -> List[Experience]:
        pass

    @abstractmethod
    def get_last(self, number: int) -> List[Experience]:
        pass

    @abstractmethod
    def getAll(self) -> List[Experience]:
        pass

    @abstractmethod
    def size(self) -> int:
        pass


class BoundedQueue(ReplayBuffer):
    def __init__(self, bufferSize: int):
        self._bufferSize = bufferSize
        self._queue = []

    def reset(self) -> None:
        self._queue = []

    def insert(self, actualState: Any, action: Any, reward: float, nextState: Any) -> None:
        self._queue.insert(0, Experience(actualState, action, nextState, reward))
        self._queue = self._queue[:self._bufferSize]

    def subsample(self, batchSize: int) -> List[Experience]:
        return random.sample(self._queue, batchSize) #Need to specify seed? On Scarlib is 42

    def get_last(self, number: int) -> List[Experience]:
        return self._queue[:number]

    def getAll(self) -> List[Experience]:
        return self._queue

    def size(self) -> int:
        return len(self._queue)


def ReplayBufferFactory(size: int) -> BoundedQueue:
    return BoundedQueue(size)