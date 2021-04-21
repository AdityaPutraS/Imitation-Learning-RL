import pybullet
from hier_env import HierarchicalHumanoidEnv
import time
import numpy as np
import pandas as pd
import ray
from ray.tune import function
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from train_config import config_hier, config_low, single_env
from collections import OrderedDict

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

if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    single_env = HierarchicalHumanoidEnv()

    
    agent = PPOTrainer(config_hier)
    experiment_name = "HWalk_Hier_Mimic"
    experiment_id = "train_HumanoidBulletEnvHier-v0_ad955_00000_0_2021-04-20_13-24-28"
    checkpoint_num = "270"
    agent.restore(
        "/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint_{}/checkpoint-{}".format(
            experiment_name, experiment_id, checkpoint_num, checkpoint_num, checkpoint_num
        )
    )

    # experiment_name = "HWalk_Low_Mimic"
    # experiment_id = "PPO_HumanoidBulletEnvLow-v0_699c9_00000_0_2021-04-18_22-14-39"
    # checkpoint_num = "1930"

    # config_low["num_workers"] = 0
    # config_low["num_envs_per_worker"] = 1
    # config_low["num_gpus"] = 1
    # agentLow = PPOTrainer(config_low)
    # agentLow.restore(
    #     "/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
    #         experiment_name, experiment_id, checkpoint_num, checkpoint_num
    #     )
    # )
    # lowWeight = agentLow.get_policy().get_weights()
    # highWeight = agent.get_policy("low_level_policy").get_weights()
    # lowState = agentLow.get_policy().get_state()
    # importedOptState = OrderedDict(
    #     [
    #         (k.replace("default_policy", "low_level_policy"), v)
    #         for k, v in lowState["_optimizer_variables"].items()
    #     ]
    # )
    # importedPolicy = {
    #     hw: lowWeight[lw] for hw, lw in zip(highWeight.keys(), lowWeight.keys())
    # }
    # importedPolicy["_optimizer_variables"] = importedOptState
    # agent.get_policy("low_level_policy").set_state(importedPolicy)

    env = HierarchicalHumanoidEnv()

    fps = 60.0
    qKey = ord("q")
    rKey = ord("r")
    eKey = ord("e")

    doneAll = False
    while not doneAll:
        done = False
        env.render()
        observation = env.reset()
        print("Start from frame: ", env.selected_motion_frame)
        # sinObs, cosObs = observation[1], observation[2]
        # degObs = np.rad2deg(np.arctan2(sinObs, cosObs))
        # print("Deg obs: ", degObs)
        
        drawAxis()
        pause = True
        while not done and not doneAll:
            action = dict()
            if(not pause):
                if('high_level_agent' in observation):
                    action['high_level_agent'] = agent.compute_action(observation['high_level_agent'], policy_id='high_level_policy')
                    # if(not pause):
                        # pause = True
                else:
                    action[env.low_level_agent_id] = agent.compute_action(observation[env.low_level_agent_id], policy_id='low_level_policy')
                observation, reward, f_done, info = env.step(action)
                done = f_done['__all__'] == True
            targetHL = np.array([
                np.cos(env.highLevelDegTarget),
                np.sin(env.highLevelDegTarget),
                0
            ]) * 5
            drawLine(env.robot_pos, env.robot_pos + targetHL, [0, 1, 0])

            drawLine(env.starting_ep_pos, env.robot_pos + targetHL, [0, 0, 1])

            time.sleep(1.0 / fps)

            keys = pybullet.getKeyboardEvents()
            if qKey in keys and keys[qKey] & pybullet.KEY_WAS_TRIGGERED:
                print("QUIT")
                doneAll = True
            elif rKey in keys and keys[rKey] & pybullet.KEY_WAS_TRIGGERED:
                done = True
            elif eKey in keys and keys[eKey] & pybullet.KEY_WAS_TRIGGERED:
                pause = not pause
        print("Survived {} steps".format(env.cur_timestep))
        pybullet.removeAllUserDebugItems()
    env.close()
    ray.shutdown()