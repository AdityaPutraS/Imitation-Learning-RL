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

from train_config import config_hier, config_low, single_env
from collections import OrderedDict

experiment_name = "HWalk_Low_Mimic"
experiment_id = "PPO_HumanoidBulletEnvLow-v0_699c9_00000_0_2021-04-18_22-14-39"
checkpoint_num = "1930"


experiment_name_hier = "HWalk_Hier_Mimic"
experiment_id_hier = "PPO_HumanoidBulletEnvHier-v0_66b0b_00000_0_2021-04-11_16-39-16"
checkpoint_num_hier = "940"

resumeFromCheckpoint = False
useModelFromLowLevelTrain = True


def train(config, reporter):
    trainer = PPOTrainer(config=config)

    if useModelFromLowLevelTrain:
        config_low["num_workers"] = 0
        config_low["num_envs_per_worker"] = 1
        config_low["num_gpus"] = 1
        agentLow = PPOTrainer(config_low)
        agentLow.restore(
            "/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
                experiment_name, experiment_id, checkpoint_num, checkpoint_num
            )
        )
        lowWeight = agentLow.get_policy().get_weights()
        highWeight = trainer.get_policy("low_level_policy").get_weights()
        lowState = agentLow.get_policy().get_state()
        importedOptState = OrderedDict(
            [
                (k.replace("default_policy", "low_level_policy"), v)
                for k, v in lowState["_optimizer_variables"].items()
            ]
        )
        importedPolicy = {
            hw: lowWeight[lw] for hw, lw in zip(highWeight.keys(), lowWeight.keys())
        }
        importedPolicy["_optimizer_variables"] = importedOptState
        trainer.get_policy("low_level_policy").set_state(importedPolicy)

    while True:
        result = trainer.train()
        reporter(**result)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    resources = PPOTrainer.default_resource_request(config_hier).to_json()
    tune.run(
        train,
        name="HWalk_Hier_Mimic",
        # resume=resume,
        restore="/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
            experiment_name_hier,
            experiment_id_hier,
            checkpoint_num_hier,
            checkpoint_num_hier,
        )
        if resumeFromCheckpoint
        else "",
        checkpoint_at_end=True,
        checkpoint_freq=10,
        checkpoint_score_attr="episode_reward_mean",
        stop={"episode_reward_mean": 10000},
        config=config_hier,
        resources_per_trial=resources,
    )
