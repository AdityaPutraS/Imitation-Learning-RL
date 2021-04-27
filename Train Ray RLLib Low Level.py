#!/usr/bin/env python
# coding: utf-8

from typing import Dict

import gym
from gym.spaces import Discrete, Tuple, Box
import pybullet_envs

import ray
from ray import tune
from ray.tune import function
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg import ApexDDPGTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.a3c import A2CTrainer, A3CTrainer
from ray.tune.registry import register_env

from low_level_env import LowLevelHumanoidEnv
from custom_callback import RewardLogCallback
from train_config import (
    config_low,
    config_low_pg,
    config_low_ddpg,
    config_low_impala,
    config_low_apex_ddpg,
    config_low_a2c,
)

# ray.shutdown()
ray.init(ignore_reinit_error=True)

TARGET_REWARD = 10000

experiment_name = "HWalk_Low_Mimic"
experiment_id = "PPO_HumanoidBulletEnv-v0-Low_807f6_00000_0_2021-04-25_08-45-16"
checkpoint_num = "2400"

resume = False

tune.run(
    PGTrainer,
    name="HWalk_Low_Mimic",
    # resume=resume,
    restore="/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
        experiment_name, experiment_id, checkpoint_num, checkpoint_num
    )
    if resume
    else "",
    checkpoint_at_end=True,
    checkpoint_freq=10,
    checkpoint_score_attr="episode_reward_mean",
    stop={"episode_reward_mean": TARGET_REWARD},
    config=config_low_pg,
)
