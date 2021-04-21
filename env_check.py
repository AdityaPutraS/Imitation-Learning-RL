import pybullet
from low_level_env import LowLevelHumanoidEnv
import time
import numpy as np
import pandas as pd
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from train_config import config_low


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

    agent = PPOTrainer(config_low)
    experiment_name = "HWalk_Low_Mimic"
    experiment_id = "PPO_HumanoidBulletEnvLow-v0_71287_00000_0_2021-04-21_08-35-16"
    checkpoint_num = "3700"
    agent.restore(
        "/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
            experiment_name, experiment_id, checkpoint_num, checkpoint_num
        )
    )

    env = LowLevelHumanoidEnv()

    fps = 60.0
    qKey = ord("q")
    rKey = ord("r")
    eKey = ord("e")

    doneAll = False
    while not doneAll:
        done = False
        env.render()
        observation = env.reset()
        print("Start from frame: ", env.frame)
        sinObs, cosObs = observation[1], observation[2]
        degObs = np.rad2deg(np.arctan2(sinObs, cosObs))
        print("Deg obs: ", degObs)
        # print(env.target, env.targetHighLevel)
        drawAxis()
        pause = True
        r, p, y = env.flat_env.robot.robot_body.pose().rpy()
        print(r, p, y)
        while not done and not doneAll:
            if (not pause):
                action = agent.compute_action(observation)
                observation, reward, f_done, info = env.step(action)
                # r, p, y = env.flat_env.robot.robot_body.pose().rpy()
                # print(r, p, y - env.highLevelDegTarget)
            # print(env.lowTargetScore)
            # Garis dari origin ke target akhir yang harus dicapai robot
            # drawLine([0, 0, 0], env.target, [0, 1, 0])
            
            robotPos = np.array(env.flat_env.robot.body_xyz)
            robotPos[2] = 0
            # Garis dari origin ke robot
            # drawLine([0, 0, 0], robotPos, [1, 1, 1])

            # Garis dari robot ke walk target environment
            vHighTarget = np.array([np.cos(env.highLevelDegTarget), np.sin(env.highLevelDegTarget), 0]) * 10
            drawLine(robotPos, robotPos + vHighTarget, [0, 0, 0])

            # drawLine(env.starting_ep_pos + np.array([0, 0, 1]), robotPos + env.targetHighLevel, [0, 0, 1])

            drawLine(robotPos, [env.flat_env.robot.walk_target_x, env.flat_env.robot.walk_target_y, 0], [0, 0, 1])
            # print(observation)
            # drawText(str(env.frame), env.flat_env.parts["lwaist"].get_position() + np.array([0, 0, 1]), [0, 1, 0], 1.0/30)
            # drawText(str(env.deltaJoints), env.flat_env.parts["lwaist"].get_position() + np.array([1, 0, 1]), [1, 0, 0], 1.0/30)
            # drawText(str(env.deltaEndPoints), env.flat_env.parts["lwaist"].get_position() + np.array([-1, 0, 1]), [0, 0, 1], 1.0/30)

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