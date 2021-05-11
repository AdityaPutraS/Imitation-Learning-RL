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
    config_low_pg,
    config_low_ddpg,
    config_low_impala,
    config_low_apex_ddpg,
    config_low_a2c,
)

from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
import random

# ray.shutdown()
ray.init(ignore_reinit_error=True)

TARGET_REWARD = 10000

experiment_name = "HWalk_Low_Mimic"
experiment_id = "PPO_HumanoidBulletEnv-v0-Low_807f6_00000_0_2021-04-25_08-45-16"
checkpoint_num = "2400"

resume = False

def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] <= config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    config["train_batch_size"] = int(config["train_batch_size"])
    return config

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
        "num_sgd_iter": lambda: random.randint(1, 30),
        "sgd_minibatch_size": lambda: random.randint(128, 4096),
        "train_batch_size": lambda: random.randint(2000, 40000),
        "gamma": lambda: random.uniform(0.8, 0.9997),
        "kl_coeff": lambda: random.uniform(0.3, 1)
    },
    custom_explore_fn=explore)

pb2 = PB2(
    time_attr='time_total_s',
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=120,
    quantile_fraction=0.25,
    hyperparam_bounds={
        "lambda": [0.9, 1.0],
        "clip_param": [0.01, 0.5],
        "lr": [1e-6, 1e-3],
        "num_sgd_iter": [1, 30],
        "sgd_minibatch_size": [128, 4096],
        "train_batch_size": [2000, 40000],
        "gamma": [0.8, 0.9997],
        "kl_coeff": [0.3, 1]
    })

analysis = tune.run(
    PPOTrainer,
    name="HWalk_Low_Mimic",
    resume=False,
    # restore="/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
    #     experiment_name, experiment_id, checkpoint_num, checkpoint_num
    # )
    # if resume
    # else "",
    checkpoint_at_end=True,
    checkpoint_freq=10,
    checkpoint_score_attr="episode_reward_mean",
    # scheduler=pb2,
    # num_samples=4,
    # stop={"timesteps_total": 5000000},
    config=config_low_best,
)

# print("best hyperparameters: ", analysis.best_config)