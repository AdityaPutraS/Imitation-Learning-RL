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
    config_low_search,
    config_low_best,
    config_low_best_2,
    config_low_pg,
    config_low_ddpg,
    config_low_impala,
    config_low_apex_ddpg,
    config_low_a2c,
)

from ray.tune.schedulers.pb2 import PB2

# ray.shutdown()
ray.init(ignore_reinit_error=True)

TARGET_REWARD = 10000

experiment_name = "HWalk_Low_Mimic_Search_3"
experiment_id = "PPO_HumanoidBulletEnv-v0-Low_4964e_00001_1_2021-05-25_10-04-17"
checkpoint_num = "349"

resume = False

pb2 = PB2(
    time_attr='training_iteration',
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=10,
    quantile_fraction=0.25,
    hyperparam_bounds={
        "lambda": [0.9, 1.0],
        "clip_param": [0.01, 0.5],
        "lr": [1e-6, 1e-3],
        # "num_sgd_iter": [3, 30],
        "train_batch_size": [8192, 40000]
        # "gamma": [0.8, 0.9997],
        # "kl_coeff": [0.3, 1],
        # "vf_loss_coeff": [0.5, 1],
        # "entropy_coeff": [0, 0.01],
    })

analysis = tune.run(
    PPOTrainer,
    name="HWalk_Low_Mimic_Search_7",
    resume=False,
    # restore="/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
    #     experiment_name, experiment_id, checkpoint_num, checkpoint_num
    # ),
    # if resume
    # else "",
    checkpoint_at_end=True,
    checkpoint_freq=10,
    checkpoint_score_attr="episode_reward_mean",
    # scheduler=pb2,
    # num_samples=5,
    # stop={"episode_reward_mean": 1000},
    config=config_low_best,
)

print("best hyperparameters: ", analysis.get_best_config("episode_reward_mean", "max"))