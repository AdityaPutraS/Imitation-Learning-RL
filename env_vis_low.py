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

from humanoid import CustomHumanoidRobot

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


    model = input("Jenis model (08_03, 09_03, custom_env, quit): ").lower()
    while(model != 'quit'):
    
        agent = PPOTrainer(config_low)
        experiment_name = "HWalk_Low_Mimic"
        
        useCustomEnv = False
        # Tanpa endpoint reward
        if(model == '08_03'):
            # Motion 08_03
            experiment_id = "PPO_HumanoidBulletEnv-v0-Low_68eec_00000_0_2021-05-01_23-32-16"
            checkpoint_num = "1610"
            motion_used = "motion08_03"
            print("Selesai load model motion 08_03")
        elif(model == 'custom_env'):
            # Custom env
            experiment_id = "PPO_HumanoidBulletEnv-v0-Low_54134_00000_0_2021-05-03_21-13-19"
            checkpoint_num = "3640"
            motion_used = "motion09_03"
            useCustomEnv = True
            print("Selesai load model custom env motion 09_03")
        elif(model == 'l'):
            experiment_name = "HWalk_Low_Mimic_Search_5"
            experiment_id = "PPO_HumanoidBulletEnv-v0-Low_80fea_00001_1_2021-05-26_12-12-47"
            checkpoint_num = "440"
            motion_used = "motion09_03"
            useCustomEnv = False
            print("Selesai load model latest")
        else:
            # Motion 09_03
            experiment_id = "PPO_HumanoidBulletEnv-v0-Low_6d114_00000_0_2021-04-30_23-26-25"
            checkpoint_num = "1690"
            motion_used = "motion09_03"
            print("Selesai load model motion 09_03")

        

        agent.restore(
            "/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
                experiment_name, experiment_id, checkpoint_num, checkpoint_num
            )
        )

        # 29_21-30-25 (motion08_03)
        # 25_19-33-42 (motion09_03)

        env = LowLevelHumanoidEnv(reference_name=motion_used, useCustomEnv=useCustomEnv, customRobot=CustomHumanoidRobot())
        env.usePredefinedTarget = True
        
        tipeTarget = input("Jenis target (kotak, m, segi_enam, 180, lurus, tanjakan): ").lower()
        if (tipeTarget == 'kotak'):
            # Kotak
            env.predefinedTarget = np.array([
                [5, 0, 0],
                [5, -5, 0],
                [0, -5, 0],
                [0, 0, 0]
            ])
        elif (tipeTarget == 'm'):
            # M
            env.predefinedTarget = np.array([
                [5, -1, 0],
                [0, -2, 0],
                [5, -3, 0],
                [0, -4, 0],
                [0, 0, 0]
            ])
        elif (tipeTarget == 'segi enam'):
            # Segi Enam
            env.predefinedTarget = np.array([
                [0, 5, 0],
                [4.33, 2.5, 0],
                [4.33, -2.5, 0],
                [0, -5, 0],
                [-4.33, -2.5, 0],
                [-4.33, 2.5, 0],
            ])
        elif (tipeTarget == '180'):
            # Putar 180 Derajat
            env.predefinedTarget = np.array([
                [0, 5, 0],
                [0, -5, 0],
                [-5, 0, 0],
                [5, 0, 0]
            ])
        elif (tipeTarget == 'lurus' or tipeTarget == 'tanjakan'):
            # Lurus 10 meter, kemudian balik
            env.predefinedTarget = np.array([
                [10, 0, 0],
                [0, 0, 0]
            ])

        fps = 240.0
        qKey = ord("q")
        rKey = ord("r")
        eKey = ord("e")

        doneAll = False
        logStarted = False
        
        # Buat tanjakan untuk custom env
        if(useCustomEnv and tipeTarget == 'tanjakan'):
            terrainData = [0] * 256 * 256
            for j in range(63-5, 64+5+1):
                for i in range(63, 68):
                    terrainData[2 * i + 2 * j * 256] = (i-63)/10
                    terrainData[2 * i + 1 + 2 * j * 256] = (i-63)/10
                    terrainData[2 * i + (2 * j + 1) * 256] = (i-63)/10
                    terrainData[2 * i + 1 + (2 * j + 1) * 256] = (i-63)/10

        while not doneAll:
            done = False
            env.render()
            observation = env.resetFromFrame(startFrame=0, startFromRef=False, initVel=False)
            if(useCustomEnv and tipeTarget == 'tanjakan'):
                env.flat_env.stadium_scene.replaceHeightfieldData(terrainData)
                print("Selesai replace heightfield")
            drawAxis()
            if(not(logStarted)):
                pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, "./out/video/{}_{}_{}_{}.mp4".format(experiment_id[-19:], checkpoint_num, model, tipeTarget))
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
    
        model = input("Jenis model (08_03, 09_03, custom_env, quit): ").lower()

    ray.shutdown()
