import pybullet
from low_level_env import LowLevelHumanoidEnv
import time
import numpy as np
import pandas as pd
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from train_config import config_low
from math_util import projPointLineSegment

from scipy.spatial.transform import Rotation as R
import json
import copy


def drawLine(c1, c2, color):
    return pybullet.addUserDebugLine(
        c1, c2, lineColorRGB=color, lineWidth=5, lifeTime=0.1
    )


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

def calcDrift(pos, lineStart, lineEnd):
    projection = projPointLineSegment(pos, lineStart, lineEnd)
    return np.linalg.norm(projection - pos)
    
if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    # Khusus 19-33-42 (2580)
    # config_low["model"]["fcnet_hiddens"] = [256, 128]

    agent = PPOTrainer(config_low)
    experiment_name = "HWalk_Low_Mimic"
    experiment_id = "PPO_HumanoidBulletEnv-v0-Low_0e400_00000_0_2021-04-29_21-30-25"
    checkpoint_num = "2190"
    # experiment_id = "PPO_HumanoidBulletEnv-v0-Low_166df_00000_0_2021-04-25_19-33-42"
    # checkpoint_num = "2580"
    agent.restore(
        "/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
            experiment_name, experiment_id, checkpoint_num, checkpoint_num
        )
    )

    # 29_21-30-25 (motion08_03)
    # 25_19-33-42 (motion09_03)
    motion_used = "motion08_03"
    env = LowLevelHumanoidEnv(reference_name=motion_used)
    env.usePredefinedTarget = True
    
    # Kotak
    # env.predefinedTarget = np.array([
    #     [5, 0, 0],
    #     [5, -5, 0],
    #     [0, -5, 0],
    #     [0, 0, 0]
    # ])

    # M
    env.predefinedTarget = np.array([
        [5, -1, 0],
        [0, -2, 0],
        [5, -3, 0],
        [0, -4, 0],
        [0, 0, 0]
    ])

    # Segi Enam
    # env.predefinedTarget = np.array([
    #     [0, 5, 0],
    #     [4.33, 2.5, 0],
    #     [4.33, -2.5, 0],
    #     [0, -5, 0],
    #     [-4.33, -2.5, 0],
    #     [-4.33, 2.5, 0],
    # ])

    # Putar 180 Derajat
    # env.predefinedTarget = np.array([
    #     [0, 5, 0],
    #     [0, -5, 0],
    #     [-5, 0, 0],
    #     [5, 0, 0]
    # ])

    fps = 240.0
    qKey = ord("q")
    rKey = ord("r")
    eKey = ord("e")

    doneAll = False
    logStarted = False
    while not doneAll:
        done = False
        env.render()
        observation = env.reset(startFrame=0, startFromRef=True)
        drawAxis()
        if(not(logStarted)):
            pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, "./out/video/{}_{}.mp4".format(experiment_id[:-19], checkpoint_num))
            logStarted = True
        pause = True

        # drift = []
        while not done and not doneAll:
            if not pause:
                action = agent.compute_action(observation)
                observation, reward, f_done, info = env.step(action)
                # done = f_done
                # drift.append(calcDrift(env.robot_pos, env.starting_robot_pos, env.target))
            
            drawLine(
                env.robot_pos,
                env.target,
                [0, 0, 1],
            )

            # drawLine(
            #     env.starting_ep_pos,
            #     env.target,
            #     [0, 0, 0],
            # )

            # drawLine(
            #     env.starting_ep_pos,
            #     env.robot_pos,
            #     [1, 1, 1]
            # )

            # time.sleep(1.0 / fps)

            keys = pybullet.getKeyboardEvents()
            if qKey in keys and keys[qKey] & pybullet.KEY_WAS_TRIGGERED:
                print("QUIT")
                doneAll = True
            elif rKey in keys and keys[rKey] & pybullet.KEY_WAS_TRIGGERED:
                done = True
            elif eKey in keys and keys[eKey] & pybullet.KEY_WAS_TRIGGERED:
                pause = not pause
        print("    Hidup selama {} steps".format(env.cur_timestep))
        pybullet.removeAllUserDebugItems()
    env.close()
    ray.shutdown()
