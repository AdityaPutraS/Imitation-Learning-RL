#!/usr/bin/env python
# coding: utf-8

import gym
from gym.spaces import Discrete, Tuple, Box
import pybullet_envs

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from hierarchical_env import HierarchicalHumanoidEnv

# ray.shutdown()
ray.init(ignore_reinit_error=True)


ENV = HierarchicalHumanoidEnv()

TARGET_REWARD = 2000


def policy_mapping_fn(agent_id):
    if agent_id.startswith("low_level_"):
        return "low_level_policy"
    else:
        return "high_level_policy"

highLevelPolicy = (
    None,
    ENV.observation_space,
    Box(),
    {
        "model": {}
    }
)

config = {
    "env": ENV,
    "num_workers": 12,
    "multiagent": {
        "policies": {
            "high_level_policy": (None),
            "low_level_policy": (None),
        },
        "policy_mapping_fn": function(policy_mapping_fn),
    },
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": 0.995,
    "lambda": 0.95,
    "clip_param": 0.2,
    "kl_coeff": 1.0,
    "num_sgd_iter": 20,
    "lr": .0001,
    "sgd_minibatch_size": 8000,
    "train_batch_size": 24000,
    "model": {
        "fcnet_hiddens": [256, 128, 64],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "complete_episodes",
    "observation_filter": "MeanStdFilter",
}

tune.run(
    PPOTrainer,
    name="HWalk_Hier",
    resume=False,
    checkpoint_at_end=True,
    checkpoint_freq=5,
    keep_checkpoints_num=50,
    checkpoint_score_attr="episode_reward_max",
    stop={"episode_reward_mean": TARGET_REWARD},
    config=config
)


# In[ ]:
