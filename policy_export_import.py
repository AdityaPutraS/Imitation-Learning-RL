import pybullet
from hier_env import HierarchicalHumanoidEnv
from low_level_env import LowLevelHumanoidEnv
import time
import numpy as np
import pandas as pd
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune import function
import pickle
from collections import OrderedDict

from train_config import config_hier, config_low, single_env

if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    agentLow = PPOTrainer(config_low)
    experiment_name = "HWalk_Low_Mimic"
    experiment_id = "PPO_HumanoidBulletEnvLow-v0_699c9_00000_0_2021-04-18_22-14-39"
    checkpoint_num = "1930"
    agentLow.restore(
        "/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
            experiment_name, experiment_id, checkpoint_num, checkpoint_num
        )
    )

    # agent.export_policy_model("out/model", "default_policy")
    # agent.import_model("out/model")

    # agent.get_policy("default_policy").import_model_from_h5

    agentHigh = PPOTrainer(config_hier)
    lowWeight = agentLow.get_policy().get_weights()
    highWeight = agentHigh.get_policy("low_level_policy").get_weights()
    importedPolicy = { hw: lowWeight[lw] for hw, lw in zip(highWeight.keys(), lowWeight.keys())}
    s1 = agentLow.get_policy().get_state()
    s11 = OrderedDict([(k.replace("default_policy", "low_level_policy"), v) for k, v in s1['_optimizer_variables'].items()])
    importedPolicy['_optimizer_variables'] = s11
    agentHigh.get_policy("low_level_policy").set_state(importedPolicy)

    obs = single_env.low_level_obs_space.sample()
    print(agentLow.compute_action(obs))
    print(agentHigh.compute_action(obs, policy_id='low_level_policy'))
    print("=============================================================")
    ray.shutdown()