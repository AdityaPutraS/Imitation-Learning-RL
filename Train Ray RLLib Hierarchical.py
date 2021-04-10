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

import numpy as np

# from low_level_env import LowLevelHumanoidEnv
from hier_env import HierarchicalHumanoidEnv
from custom_callback import RewardLogCallback

# ray.shutdown()
ray.init(ignore_reinit_error=True)

single_env = HierarchicalHumanoidEnv()


def make_env(env_config):
    import pybullet_envs

    return HierarchicalHumanoidEnv()


ENV_NAME = "HumanoidBulletEnvHier-v0"
register_env(ENV_NAME, make_env)
TARGET_REWARD = 5000


def policy_mapping_fn(agent_id):
    if agent_id.startswith("low_level_"):
        return "low_level_policy"
    else:
        return "high_level_policy"


highLevelPolicy = (
    None,
    single_env.high_level_obs_space,
    single_env.high_level_act_space,
    {
        "model": {
            "fcnet_hiddens": [512, 256],
            "fcnet_activation": "tanh",
            "free_log_std": False,
            # "use_lstm": True,
            # "max_seq_len": 25,
            # "lstm_use_prev_action": True,
            # "lstm_use_prev_reward": True,
        },
    },
)

lowLevelPolicy = (
    None,
    single_env.low_level_obs_space,
    single_env.low_level_act_space,
    {
        "model": {
            "fcnet_hiddens": [1024, 512],
            "fcnet_activation": "tanh",
            "free_log_std": True,
        },
    },
)

config = {
    "env": ENV_NAME,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 20,
    "multiagent": {
        "policies": {
            "high_level_policy": highLevelPolicy,
            "low_level_policy": lowLevelPolicy,
        },
        "policy_mapping_fn": function(policy_mapping_fn),
    },
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
    "sgd_minibatch_size": 12000,
    "train_batch_size": 36000,
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
}

experiment_name = "HWalk_Hier_Mimic"
experiment_id = "PPO_HumanoidBulletEnvHier-v0_02a53_00000_0_2021-04-10_06-57-50"
checkpoint_num = "550"

resume = True

tune.run(
    PPOTrainer,
    name="HWalk_Hier_Mimic",
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
