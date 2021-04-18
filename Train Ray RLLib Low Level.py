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
from ray.tune.registry import register_env

from low_level_env import LowLevelHumanoidEnv
from custom_callback import RewardLogCallback

# ray.shutdown()
ray.init(ignore_reinit_error=True)

single_env = LowLevelHumanoidEnv()
ENV_NAME = "HumanoidBulletEnvLow-v0"

def make_env(env_config):
    import pybullet_envs
    return LowLevelHumanoidEnv()

register_env(ENV_NAME, make_env)
TARGET_REWARD = 10000


def policy_mapping_fn(agent_id):
    return "low_level_policy"


config = {
    "env": ENV_NAME,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 20,
    "log_level": "WARN",
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": 0.995,
    "lambda": 0.95,
    "clip_param": 0.2,
    "kl_coeff": 1.0,
    "num_sgd_iter": 20,
    "lr": 0.0005,
    "vf_clip_param": 10,
    "sgd_minibatch_size": 12000,
    "train_batch_size": 36000,
    "model": {
        "fcnet_hiddens": [1024, 512],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "framework": "tf",
}

experiment_name = "HWalk_Low_Mimic"
experiment_id = "PPO_HumanoidBulletEnvLow-v0_f8215_00000_0_2021-04-18_16-56-31"
checkpoint_num = "930"

resume = False

tune.run(
    PPOTrainer,
    name="HWalk_Low_Mimic",
    # resume=resume,
    restore="/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
        experiment_name, experiment_id, checkpoint_num, checkpoint_num
    ) if resume else "",
    checkpoint_at_end=True,
    checkpoint_freq=10,
    checkpoint_score_attr="episode_reward_mean",
    stop={"episode_reward_mean": TARGET_REWARD},
    config=config,
)
