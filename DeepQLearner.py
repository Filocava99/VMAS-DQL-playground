from datetime import datetime

import numpy
import torch
from torch import optim, Tensor
import random
from typing import List, Type

from torch.nn import SmoothL1Loss

import Device
from DQN import SimpleSequentialDQN
from LearningConfiguration import LearningConfiguration
from ReplayBuffer import ReplayBuffer
import Device
from Cleaning import Scenario as CleaningScenario


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
        self.device = torch.device(Device.get())
        self.target_network = learning_configuration.dqn_factory.createNN()
        self.policy_network = learning_configuration.dqn_factory.createNN()
        self.targetPolicy = DeepQLearner.policy_from_network(self.policy_network, action_space)
        self.behaviouralPolicy = DeepQLearner.policy_from_network(self.policy_network, action_space)
        self.optimizer = optim.Adam(self.policy_network.parameters(), self.learning_rate)
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
            sequence = self.prepare_sequence(state)
            return self.behaviouralPolicy(sequence)

    def prepare_sequence(self, state) -> Tensor:
        memory = self.memory.get_last(50)
        sequence = []
        for i in range(len(memory)):
            tensor = torch.cat((memory[i].actualState, memory[i].action, torch.tensor([memory[i].reward])), dim=-1)
            sequence.append(tensor)
        sequence.append(torch.cat((state, torch.tensor([0.0, 0.0]), torch.tensor([0.0])), dim=-1))
        batch = torch.stack(sequence)
        n_env = 1
        result = batch.view([n_env, len(batch), len(batch[0])])
        return result

    def get_random_contiguous_sequence(self, n):
        lst = self.memory.getAll()
        if len(lst) < n:
            print("N is greater than the list length!")
            return
        idx = random.randint(0, len(lst) - n)
        indices = [idx]
        for _ in range(n-1):
            idx = random.randint(idx, len(lst) - n + len(indices))
            indices.append(idx)
        return [lst[i] for i in indices]

    def prepare_random_sequence(self, n=50):
        sequence = []
        for _ in range(self.batch_size):
            sequence.append(self.get_random_contiguous_sequence(n))
        # sequence = []
        # for i in indices:
        #     tensor = torch.cat((self.memory.getAll()[i].actualState, self.memory.getAll()[i].action, torch.tensor([self.memory.getAll()[i].reward])), dim=-1)
        #     sequence.append(tensor)
        # batch = torch.stack(sequence)
        # n_env = 1
        # result = batch.view([n_env, len(batch), len(batch[0])])
        return sequence

    def improve(self):
        memory_sample = self.prepare_random_sequence()
        if len(memory_sample) == self.batch_size:
            states = torch.stack([torch.stack([torch.cat([x.actualState, x.action, x.reward], dim=-1) for x in sublist]) for sublist in memory_sample]).to(Device.get())
            actions_list = [[x.action for x in sublist] for sublist in memory_sample]
            # actions = torch.stack(actions_list)
            rewards = torch.stack([torch.stack([x.reward for x in sublist]) for sublist in memory_sample]).to(Device.get())
            next_states = torch.stack([torch.stack([torch.cat([x.nextState, x.action, x.reward], dim=-1) for x in sublist]) for sublist in memory_sample]).to(Device.get())
            actions_indexes = []
            for j, action_sublists in enumerate(actions_list):
                seq = []
                for action in action_sublists:
                    for i in range(len(self.action_space)):
                        if torch.equal(action, self.action_space[i]):
                            seq.append(torch.tensor(i, dtype=torch.long))
                            break
                actions_indexes.append(torch.stack(seq, dim=-1).to(Device.get()))
            actions_indexes = torch.stack(actions_indexes, dim=0).to(Device.get())
            state_action_value = self.policy_network.forward(states)
            state_action_value = state_action_value.gather(dim=1, index=actions_indexes)
            max_next_state_values = self.target_network.forward(next_states).max(1)
            indices = max_next_state_values[1].detach()
            next_state_values = max_next_state_values[0].detach()
            expected_value = (next_state_values.view(-1, 1) * self.gamma) + rewards[indices].max(1)[0]
            criterion = SmoothL1Loss()
            loss = criterion(state_action_value.max(1)[0].view(-1, 1), expected_value)
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
        torch.save(self.target_network.state_dict(),
                   f"{CleaningScenario.__class__.__name__}-{episode}-{time_mark}-agent-{agent_id}")

    def load_snapshot(self, path):
        self.target_network.load_state_dict(torch.load(path))

    def load_snapshot(self, path):
        self.target_network.load_state_dict(torch.load(path))
        self.policy_network.load_state_dict(torch.load(path))

    @staticmethod
    def policy_from_network(network, action_space):
        def _policy(state):
            with torch.no_grad():
                action_index = network.forward(state).max(dim=-1)[1].item()
                return action_space[action_index]

        return _policy


class SimpleSequentialDQNFactory:
    @staticmethod
    def create_NN():
        return SimpleSequentialDQN()
