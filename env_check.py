import pybullet
from low_level_env import LowLevelHumanoidEnv
import time
import numpy as np
import pandas as pd
import ray
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


def make_env_low(env_config):
    import pybullet_envs

    return LowLevelHumanoidEnv()


if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    ENV_LOW = "HumanoidBulletEnv-v0-Low"
    register_env(ENV_LOW, make_env_low)
    config_low = {
        "env": ENV_LOW,
        "num_workers": 0,
        "num_envs_per_worker": 1,
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
        "sgd_minibatch_size": 8000,
        "train_batch_size": 24000,
        "model": {
            "fcnet_hiddens": [1024, 512],
            "fcnet_activation": "tanh",
            "free_log_std": True,
        },
        "batch_mode": "complete_episodes",
        "observation_filter": "NoFilter",
        "framework": "tf",
    }

    agent = PPOTrainer(config_low)
    experiment_name = "HWalk_Low_Mimic"
    experiment_id = "PPO_HumanoidBulletEnvLow-v0_94516_00000_0_2021-04-14_16-44-09"
    checkpoint_num = "2250"
    agent.restore(
        "/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(
            experiment_name, experiment_id, checkpoint_num, checkpoint_num
        )
    )

    env = LowLevelHumanoidEnv()

    fps = 60.0
    qKey = ord("q")
    rKey = ord("r")

    doneAll = False
    while not doneAll:
        done = False
        env.render()
        observation = env.reset()
        sinObs, cosObs = observation[1], observation[2]
        degObs = np.rad2deg(np.arctan2(sinObs, cosObs))
        print("Deg obs: ", degObs)
        # print(env.target, env.targetHighLevel)
        drawAxis()
        while not done and not doneAll:
            action = agent.compute_action(observation)
            observation, reward, done, info = env.step(action)
            # print(env.lowTargetScore)
            # Garis dari origin ke target akhir yang harus dicapai robot
            # drawLine([0, 0, 0], env.target, [0, 1, 0])
            
            robotPos = np.array(env.flat_env.robot.body_xyz)
            robotPos[2] = 0
            # Garis dari origin ke robot
            # drawLine([0, 0, 0], robotPos, [1, 1, 1])

            # Garis dari robot ke walk target environment
            drawLine(robotPos, robotPos + env.targetHighLevel, [0, 0, 0])

            # drawLine(env.starting_ep_pos + np.array([0, 0, 1]), robotPos + env.targetHighLevel, [0, 0, 1])

            # drawLine(robotPos, [env.flat_env.robot.walk_target_x, env.flat_env.robot.walk_target_y, 0], [0, 0, 1])
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
        print("Survived {} steps".format(env.cur_timestep))
        pybullet.removeAllUserDebugItems()
    env.close()
    ray.shutdown()