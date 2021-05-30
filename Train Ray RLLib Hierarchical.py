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
from ray.tune import CLIReporter

import os

# experiment_name = "HWalk_Low_Mimic"
# experiment_id = "PPO_HumanoidBulletEnv-v0-Low_1513b_00000_0_2021-05-11_15-04-09"
# checkpoint_num = "5520"
experiment_name = "HWalk_Low_Mimic_Search_7"
experiment_id = "PPO_HumanoidBulletEnv-v0-Low_baa74_00000_0_2021-05-29_19-26-36"
checkpoint_num = "9320"


experiment_name_hier = "HWalk_Hier_Mimic_7"
experiment_id_hier = "train_HumanoidBulletEnvHier-v0_83822_00000_0_2021-05-30_14-37-33"
checkpoint_num_hier = "1"

# Cara train:
# Load model low level terlebih dahulu dengan set resumeFromCheckpoint=False,  useModelFromLowLevelTrain=True
# Tunggu hingga ke save 1 iterasi
# Train ulang dengan resumeFromCheckpoint=True,  useModelFromLowLevelTrain=False
resumeFromCheckpoint = True
useModelFromLowLevelTrain = False


def train(config, checkpoint_dir=None):
    trainer = PPOTrainer(config=config)

    if checkpoint_dir:
        trainer.load_checkpoint(checkpoint_dir)
        
    chk_freq = 10
    
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
        chk_freq = 1 # Hanya perlu 1 kali saja di awal untuk save model hasil import   

    
    while True:
        result = trainer.train()
        tune.report(**result)
        if(trainer._iteration % chk_freq == 0):
            with tune.checkpoint_dir(step=trainer._iteration) as checkpoint_dir:
                trainer.save(checkpoint_dir)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    config_hier["multiagent"]["policies_to_train"] = ["high_level_policy"]
    resources = PPOTrainer.default_resource_request(config_hier).to_json()
    tune.run(
        train,
        name="HWalk_Hier_Mimic_7",
        # resume=resume,
        restore="/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint_{}/checkpoint-{}".format(
            experiment_name_hier,
            experiment_id_hier,
            checkpoint_num_hier,
            checkpoint_num_hier,
            checkpoint_num_hier,
        )
        if resumeFromCheckpoint
        else "",
        stop={"episode_reward_mean": 10000},
        config=config_hier,
        resources_per_trial=resources
    )
