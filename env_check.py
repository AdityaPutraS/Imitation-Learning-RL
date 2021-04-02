import pybullet
from low_level_env import LowLevelHumanoidEnv
import time
import numpy as np
import pandas as pd
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

def drawLine(c1, c2, color):
    return pybullet.addUserDebugLine(c1, c2, lineColorRGB=color, lineWidth=5)

def drawAxis():
    xId = pybullet.addUserDebugLine([0, 0, 0], [10, 0, 0], lineColorRGB=[1, 0, 0], lineWidth=2)
    yId = pybullet.addUserDebugLine([0, 0, 0], [0, 5, 0], lineColorRGB=[0, 1, 0], lineWidth=2)
    zId = pybullet.addUserDebugLine([0, 0, 0], [0, 0, 15], lineColorRGB=[0, 0, 1], lineWidth=2)
    return xId, yId, zId

def drawText(text, pos, color, lifeTime):
    return pybullet.addUserDebugText(text, pos, textColorRGB=color, textSize=2, lifeTime=lifeTime)

def make_env_low(env_config):
    import pybullet_envs
    return LowLevelHumanoidEnv()

if __name__ == '__main__':
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    ENV_LOW = 'HumanoidBulletEnv-v0-Low'
    register_env(ENV_LOW, make_env_low)
    config_low = {
        "env": ENV_LOW,
        "num_workers": 0,
        "log_level": "WARN",
        "num_gpus": 1,
        "monitor": True,
        "evaluation_num_episodes": 50,
        "gamma": 0.995,
        "lambda": 0.95,
        "clip_param": 0.2,
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": 0.0001,
        "sgd_minibatch_size": 8000,
        "train_batch_size": 24000,
        "model": {
            "fcnet_hiddens": [512, 256, 128, 64],
            "fcnet_activation": "tanh",
            "free_log_std": True,
        },
        "batch_mode": "complete_episodes",
        "observation_filter": "MeanStdFilter",
    }
    
    agent = PPOTrainer(config_low)
    experiment_name = "HWalk_Low_Mimic"
    experiment_id = "PPO_HumanoidBulletEnvLow-v0_9228e_00000_0_2021-04-02_19-28-39"
    checkpoint_num = "940"
    agent.restore("/home/aditya/ray_results/{}/{}/checkpoint_{}/checkpoint-{}".format(experiment_name, experiment_id, checkpoint_num, checkpoint_num))

    env = LowLevelHumanoidEnv()

    fps = 60.0
    qKey = ord('q')
    rKey = ord('r')

    doneAll = False
    while(not doneAll):
        done = False
        env.render()
        observation = env.reset()
        pybullet.removeAllUserDebugItems()
        drawAxis()
        while(not done and not doneAll):
            action = agent.compute_action(observation)
            observation, reward, done, info = env.step(action)
            print(observation)
            # drawText(str(env.frame), env.flat_env.parts["lwaist"].get_position() + np.array([0, 0, 1]), [0, 1, 0], 1.0/30)
            # drawText(str(env.deltaJoints), env.flat_env.parts["lwaist"].get_position() + np.array([1, 0, 1]), [1, 0, 0], 1.0/30)
            # drawText(str(env.deltaEndPoints), env.flat_env.parts["lwaist"].get_position() + np.array([-1, 0, 1]), [0, 0, 1], 1.0/30)
            # TODO: visualisasikan frame yang sebenarnya
            
            time.sleep(1.0/fps)

            keys = pybullet.getKeyboardEvents()
            if qKey in keys and keys[qKey] & pybullet.KEY_WAS_TRIGGERED:
                print('QUIT')
                doneAll = True
            elif rKey in keys and keys[rKey] & pybullet.KEY_WAS_TRIGGERED:
                done = True
    env.close()
    ray.shutdown()