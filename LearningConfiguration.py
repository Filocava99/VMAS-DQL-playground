from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

from DQN import SimpleSequentialDQN


class ExponentialDecay:
    start: float
    end: float
    decay_rate: float

    def __init__(self, start, bound, decay_rate):
        self.start = start
        self.bound = bound
        self.decay_rate = decay_rate
        self.elapsed_time = 0

    def value(self):
        v = self.start * ((1-self.decay_rate)**self.elapsed_time)
        if v > self.bound:
            return v
        else:
            return self.bound

    def update(self):
        self.elapsed_time += 1


class DQNAbstractFactory(ABC):
    @abstractmethod
    def createNN(self) -> Any:
        pass

class NNFactory(DQNAbstractFactory):
    def __init__(self, input, hidden, output):
        self.input = input
        self.hidden = hidden
        self.output = output

    def createNN(self):
        return SimpleSequentialDQN(self.input, self.hidden, self.output)

@dataclass
class LearningConfiguration:
    epsilon: ExponentialDecay = ExponentialDecay(0.9, 0.01, 0.1)
    gamma: float = 0.9
    learning_rate: float = 0.0005
    batch_size: int = 32
    update_each: int = 100
    random: int = 1
    dqn_factory: DQNAbstractFactory = None
    snapshot_path: str = ""