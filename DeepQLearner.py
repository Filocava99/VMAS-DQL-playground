from datetime import datetime

import numpy
import torch
from torch import optim, Tensor
import random
from typing import List, Type

from torch.nn import SmoothL1Loss

from DQN import SimpleSequentialDQN
from LearningConfiguration import LearningConfiguration
from ReplayBuffer import ReplayBuffer


class DeepQLearner:
    def __init__(self, memory: ReplayBuffer, action_space: List[Tensor], learning_configuration: LearningConfiguration):
        self.memory = memory
        self.action_space = action_space
        self.learning_configuration = learning_configuration
        self.learning_rate = learning_configuration.learning_rate
        self.epsilon = learning_configuration.epsilon
        self.batch_size = learning_configuration.batch_size
        self.gamma = learning_configuration.gamma
        self.update_each = learning_configuration.update_each
        self.updates = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else "mps"
        if torch.backends.mps.is_available() else 'cpu')
        self.target_network = learning_configuration.dqn_factory.createNN()
        self.policy_network = learning_configuration.dqn_factory.createNN()
        self.targetPolicy = DeepQLearner.policy_from_network(self.policy_network, action_space)
        self.behaviouralPolicy = DeepQLearner.policy_from_network(self.policy_network, action_space)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(), self.learning_rate)
        self.last_loss = 0

    def record(self, state, action, reward, next_state):
        self.memory.insert(state, action, reward, next_state)

    def optimal(self, state):
        return self.targetPolicy(state)

    def behavioural(self, state):
        if random.random() < self.epsilon.value():
            random_sample = random.sample(self.action_space, len(self.action_space))[0]
            return random_sample
        else:
            return self.behaviouralPolicy(state)

    def improve(self):
        memory_sample = self.memory.subsample(self.batch_size)
        if len(memory_sample) == self.batch_size:
            states = torch.stack(list(map(lambda x: x.actualState, memory_sample)))
            actions_list = list(map(lambda x: x.action, memory_sample))
            # actions = torch.stack(actions_list)
            rewards = torch.stack(list(map(lambda x: x.reward, memory_sample)))
            next_states = torch.stack(list(map(lambda x: x.nextState, memory_sample)))
            actions_indexes = []
            for action in actions_list:
                for i in range(len(self.action_space)):
                    if torch.equal(action, self.action_space[i]):
                        actions_indexes.append(i)
                        break
            actions_indexes = torch.tensor(actions_indexes).long()# .view(-1, 1)
            state_action_value = self.policy_network.forward(states)# .gather(dim=1, index=actions)
            state_action_value = state_action_value.gather(dim=1, index=actions_indexes.unsqueeze(1)) # TODO È giusto? Lo ha suggerito copilot lol
            next_state_values = self.target_network.forward(next_states).max(1)[0].detach()
            expected_value = (next_state_values.view(-1, 1) * self.gamma) + rewards
            criterion = SmoothL1Loss()
            loss = criterion(state_action_value, expected_value)
            loss.backward()
            self.last_loss = loss.item()
            torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 1.0)
            self.optimizer.step()
            self.updates += 1
            if self.updates % self.update_each == 0:
                print("Updating target network")
                self.target_network.load_state_dict(self.policy_network.state_dict())

    def snapshot(self, episode, agent_id):
        time_mark = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        #torch.save(self.target_network.state_dict(),
                   #f"{self.learning_configuration.snapshot_path}-{episode}-{time_mark}-agent-{agent_id}")

    @staticmethod
    def policy_from_network(network, action_space):
        def _policy(state):
            with torch.no_grad():
                tensor = state.view(1, len(state))
                action_index = network.forward(tensor).max(dim=-1)[1].item()
                return action_space[action_index]
        return _policy


class SimpleSequentialDQNFactory:
    @staticmethod
    def create_NN():
        return SimpleSequentialDQN()
