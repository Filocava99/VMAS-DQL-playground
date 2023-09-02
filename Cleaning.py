import random
import typing
from typing import Dict, Callable, List

import torch
import wandb
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, X, Y, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.wandb = kwargs.get("wandb", None)
        self.n_agents = kwargs.get("n_agents", 5)
        self.n_targets = kwargs.get("n_targets", 7)
        self.active_targets = torch.full((batch_dim, 1), self.n_targets, device=device)
        self.target_distance = kwargs.get("target_distance", 0.25)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.2)
        self._lidar_range = kwargs.get("lidar_range", 10)
        self._covering_range = kwargs.get("covering_range", 0.25)
        self._agents_per_target = kwargs.get("agents_per_target", 1)
        self.targets_respawn = kwargs.get("targets_respawn", True)
        self.shared_reward = kwargs.get("shared_reward", False)
        self.n_targets_per_env = torch.full((batch_dim, 1), self.n_targets, device=device)

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", 0)
        self.covering_rew_coeff = kwargs.get("covering_rew_coeff", 1.0)
        self.time_penalty = kwargs.get("time_penalty", 0)

        self._comms_range = self._lidar_range
        self.min_collision_distance = 0.005
        self.agent_radius = 0.05
        self.target_radius = self.agent_radius

        self.viewer_zoom = 1
        self.target_color = Color.GREEN

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=1,
            y_semidim=1,
            collision_force=500,
            substeps=2,
            drag=0.25,
        )
        # Add agents
        entity_filter_agents: Callable[[Entity], bool] = lambda e: e.name.startswith(
            "agent"
        )
        entity_filter_targets: Callable[[Entity], bool] = lambda e: e.name.startswith(
            "target"
        )
        for i in range(self.n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                shape=Sphere(radius=self.agent_radius),
                sensors=[
                    Lidar(
                        world,
                        angle_start=0.05,
                        angle_end=2 * torch.pi + 0.05,
                        n_rays=15,
                        max_range=self._lidar_range,
                        entity_filter=entity_filter_agents,
                        render_color=Color.BLUE,
                    ),
                    Lidar(
                        world,
                        n_rays=15,
                        max_range=self._lidar_range,
                        entity_filter=entity_filter_targets,
                        render_color=Color.GREEN,
                    ),
                ],
            )
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            agent.covering_reward = agent.collision_rew.clone()
            world.add_agent(agent)

        self._targets = []
        for i in range(self.n_targets):
            target = Landmark(
                name=f"target_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=self.target_radius),
                color=self.target_color,
            )
            world.add_landmark(target)
            self._targets.append(target)

        self.covered_targets = torch.zeros(batch_dim, self.n_targets, device=device)
        self.shared_covering_rew = torch.zeros(batch_dim, device=device)

        return world

    def reset_world_at(self, env_index: int = None):
        placable_entities = self._targets[: self.n_targets] + self.world.agents
        if env_index is None:
            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets), False, device=self.world.device
            )
        else:
            self.all_time_covered_targets[env_index] = False
        ScenarioUtils.spawn_entities_randomly(
            entities=placable_entities,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=self._min_dist_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
        )
        for target in self._targets[self.n_targets:]:
            target.set_pos(self.get_outside_pos(env_index), batch_index=env_index)


    def respawn_targets(self, agent: Agent):
        targets = self._targets
        targets_positions = torch.stack([t.state.pos for t in targets], dim=0)
        distances = []
        expanded_agent_pos = agent.state.pos.unsqueeze(0)
        distance = torch.norm(targets_positions - expanded_agent_pos, dim=-1).unsqueeze(-1)
        for (i, target) in enumerate(targets):
            for j in range(self.world.batch_dim):
                if distance[i][j] < self.target_distance:
                    target.set_pos(torch.tensor([-10000, -10000]), batch_index=j)
                    self.active_targets[j] -= 1
                    # bool_check = self.active_targets.flatten() == 0
                    # if bool_check.all(True):
                    #     self.done()
                    #     return

    def random_pos(self):
        x_coord = torch.full((self.world.batch_dim, 1), random.uniform(-self.world.x_semidim, self.world.x_semidim))
        y_coord = torch.full((self.world.batch_dim, 1), random.uniform(-self.world.y_semidim, self.world.y_semidim))
        tensor = torch.cat((x_coord, y_coord), dim=1)
        return tensor

    def reward(self, agent: Agent):
        # return self.reward_pos(agent)
        return self.reward_lidar(agent)

    def reward_lidar(self, agent: Agent):
        targets_lidar = agent.sensors[0].measure()
        agents_lidar = agent.sensors[1].measure()
        #get distances from nearest target
        targets_positions = torch.stack([t.state.pos for t in self._targets], dim=0)
        min_distances = torch.norm(targets_positions - agent.state.pos.unsqueeze(0), dim=-1).unsqueeze(-1)
        t = min_distances.min(dim=0).values.float()

        # reward for targets
        min_distances = targets_lidar.min(dim=1, keepdim=True).values.float()
        mask = min_distances < self._lidar_range
        min_distances[~mask] = -t[~mask]
        temp = min_distances.float().clone()
        temp[mask] = self._lidar_range
        temp[mask] /= min_distances[mask]
        targets_reward = temp

        # reward for agents
        # min_distances = agents_lidar.min(dim=1, keepdim=True).values.float()
        # mask = min_distances < self._lidar_range
        # temp = min_distances.clone()
        # temp[~mask] = -self._lidar_range/3.0
        # temp[~mask] /= min_distances[~mask]
        # min_distances[~mask] = temp[~mask]
        # temp = min_distances.float().clone()
        # temp[mask] = -self._lidar_range/3.0
        # temp[mask] /= min_distances[mask]
        # agents_reward = temp

        self.respawn_targets(agent)
        final_reward = targets_reward
        # print(agent.name)
        # print(targets_reward)
        # print(agents_reward)
        # print(final_reward)
        for i in range(self.world.batch_dim):
            if self.active_targets[i] == 0:
                final_reward[i] = 0
        return final_reward

    def get_outside_pos(self, env_index):
        return torch.empty(
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p),
            device=self.world.device,
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)

    def agent_reward(self, agent):
        agent_index = self.world.agents.index(agent)

        agent.covering_reward[:] = 0
        targets_covered_by_agent = (
                self.agents_targets_dists[:, agent_index] < self._covering_range
        )
        num_covered_targets_covered_by_agent = (
                targets_covered_by_agent * self.covered_targets
        ).sum(dim=-1)
        agent.covering_reward += (
                num_covered_targets_covered_by_agent * self.covering_rew_coeff
        )
        return agent.covering_reward

    def observation(self, agent: Agent):
        lidar_1_measures = agent.sensors[0].measure()
        lidar_2_measures = agent.sensors[1].measure()
        return torch.cat(
            [
                agent.state.pos,  # 2
                agent.state.vel,  # 2
                lidar_1_measures,  # 15
                lidar_2_measures,  # 15
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "covering_reward": agent.covering_reward
            if not self.shared_reward
            else self.shared_covering_rew,
            "collision_rew": agent.collision_rew,
            "targets_covered": self.covered_targets.sum(-1),
        }
        return info

    def done(self):
        return self.all_time_covered_targets.all(dim=-1)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        for i, target in enumerate(self._targets):
            range_circle = rendering.make_circle(self._covering_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*self.target_color.value)
            geoms.append(range_circle)
        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self._comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms
