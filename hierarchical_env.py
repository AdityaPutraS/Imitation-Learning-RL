import gym
import pybullet_envs
from gym.spaces import Box, Discrete, Tuple
import logging
import random

from ray.rllib.env import MultiAgentEnv

logger = logging.getLogger(__name__)


class HierarchicalHumanoidEnv(MultiAgentEnv):
  def __init__(self, step_per_level):
        self.flat_env = gym.make('HumanoidBulletEnv-v0')
        self.action_space = self.flat_env.action_space
        self.observation_space = self.flat_env.observation_space
        self.step_per_level = step_per_level

  def reset(self):
      self.cur_obs = self.flat_env.reset()
      self.current_goal = None
      self.steps_remaining_at_level = None
      self.num_high_level_steps = 0
      # current low level agent id. This must be unique for each high level
      # step since agent ids cannot be reused.
      self.low_level_agent_id = "low_level_{}".format(
          self.num_high_level_steps)
      return {
          "high_level_agent": self.cur_obs,
      }

  def step(self, action_dict):
      assert len(action_dict) == 1, action_dict
      if "high_level_agent" in action_dict:
          return self._high_level_step(action_dict["high_level_agent"])
      else:
          return self._low_level_step(list(action_dict.values())[0])

  def _high_level_step(self, action):
      logger.debug("High level agent sets goal".format(action))
      self.current_goal = action
      self.steps_remaining_at_level = self.step_per_level
      self.num_high_level_steps += 1
      self.low_level_agent_id = "low_level_{}".format(
          self.num_high_level_steps)
      obs = {self.low_level_agent_id: [self.cur_obs, self.current_goal]}
      rew = {self.low_level_agent_id: 0}
      done = {"__all__": False}
      return obs, rew, done, {}

  def _low_level_step(self, action):
      logger.debug("Low level agent step {}".format(action))
      self.steps_remaining_at_level -= 1
      
      # Step in the actual env
      f_obs, f_rew, f_done, _ = self.flat_env.step(action)

      self.cur_obs = f_obs

      # Calculate low-level agent observation and reward
      obs = {self.low_level_agent_id: [f_obs, self.current_goal]}
      if new_pos != cur_pos:
          if new_pos == goal_pos:
              rew = {self.low_level_agent_id: 1}
          else:
              rew = {self.low_level_agent_id: -1}
      else:
          rew = {self.low_level_agent_id: 0}

      # Handle env termination & transitions back to higher level
      done = {"__all__": False}
      if f_done:
          done["__all__"] = True
          logger.debug("high level final reward {}".format(f_rew))
          rew["high_level_agent"] = f_rew
          obs["high_level_agent"] = f_obs
      elif self.steps_remaining_at_level == 0:
          done[self.low_level_agent_id] = True
          rew["high_level_agent"] = 0
          obs["high_level_agent"] = f_obs

      return obs, rew, done, {}
