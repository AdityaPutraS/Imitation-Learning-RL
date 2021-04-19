import pybullet
from hier_env import HierarchicalHumanoidEnv
import time
import numpy as np
import pandas as pd
import ray
from ray.tune import function
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env


def drawLine(c1, c2, color):
    return pybullet.addUserDebugLine(c1, c2, lineColorRGB=color, lineWidth=5, lifeTime=0.1)


def drawAxis():
    xId = pybullet.addUserDebugLine(
        [0, 0, 0], [10, 0, 0], lineColorRGB=[1, 0, 0], lineWidth=2
    )
    yId = pybullet.addUserDebugLine(
        [0, 0, 0], [0, 5, 0], lineColorRGB=[0, 1, 0], lineWidth=2
    )
    zId = pybullet.addUserDebugLine(
        [0, 0, 0], [0, 0, 15], lineColorRGB=[0, 0, 1], lineWidth=2
    )
    return xId, yId, zId


def drawText(text, pos, color, lifeTime):
    return pybullet.addUserDebugText(
        text, pos, textColorRGB=color, textSize=2, lifeTime=lifeTime
    )


def make_env_hier(env_config):
    import pybullet_envs

    return HierarchicalHumanoidEnv()

def policy_mapping_fn(agent_id):
    if agent_id.startswith("low_level_"):
        return "low_level_policy"
    else:
        return "high_level_policy"

if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    single_env = HierarchicalHumanoidEnv()

    ENV_HIER = "HumanoidBulletEnvHier-v0"
    register_env(ENV_HIER, make_env_hier)
    highLevelPolicy = (
        None,
        single_env.high_level_obs_space,
        single_env.high_level_act_space,
        {
            "model": {
                "fcnet_hiddens": [512, 256],
                "fcnet_activation": "tanh",
                "free_log_std": False,
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
        "env": ENV_HIER,
        "num_workers": 0,
        "num_envs_per_worker": 1,
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

    agent = PPOTrainer(config)
    experiment_name = "HWalk_Hier_Mimic"
    experiment_id = "PPO_HumanoidBulletEnvHier-v0_d6128_00000_0_2021-04-12_06-32-45"
    checkpoint_num = "1250"
    agent.restore(
        "/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
            experiment_name, experiment_id, checkpoint_num, checkpoint_num
        )
    )

    env = HierarchicalHumanoidEnv()

    fps = 60.0
    qKey = ord("q")
    rKey = ord("r")

    doneAll = False
    while not doneAll:
        done = False
        env.render()
        observation = env.reset()
        print("Start from frame: ", env.frame)
        sinObs, cosObs = observation[1], observation[2]
        degObs = np.rad2deg(np.arctan2(sinObs, cosObs))
        print("Deg obs: ", degObs)
        pybullet.removeAllUserDebugItems()
        drawAxis()
        while not done and not doneAll:
            action = dict()
            if('high_level_agent' in observation):
                action['high_level_agent'] = agent.compute_action(observation['high_level_agent'], policy_id='high_level_policy')
            else:
                action[env.low_level_agent_id] = agent.compute_action(observation[env.low_level_agent_id], policy_id='low_level_policy')
            observation, reward, f_done, info = env.step(action)
            
            targetHL = np.array([
                np.cos(env.highLevelDegTarget),
                np.sin(env.highLevelDegTarget),
                0
            ]) * 5
            drawLine(env.robot_pos, env.robot_pos + targetHL, [0, 0, 0])

            time.sleep(1.0 / fps)

            keys = pybullet.getKeyboardEvents()
            if qKey in keys and keys[qKey] & pybullet.KEY_WAS_TRIGGERED:
                print("QUIT")
                doneAll = True
            elif rKey in keys and keys[rKey] & pybullet.KEY_WAS_TRIGGERED:
                done = True
        print("Survived {} steps".format(env.cur_timestep))
    env.close()
    ray.shutdown()