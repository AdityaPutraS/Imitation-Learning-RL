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
    experiment_id = "PPO_HumanoidBulletEnvHier-v0_48c1b_00000_0_2021-04-10_13-40-40"
    checkpoint_num = "1220"
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
        pybullet.removeAllUserDebugItems()
        drawAxis()
        while not done and not doneAll:
            action = dict()
            if('high_level_agent' in observation):
                action['high_level_agent'] = agent.compute_action(observation['high_level_agent'], policy_id='high_level_policy')
            else:
                action[env.low_level_agent_id] = agent.compute_action(observation[env.low_level_agent_id], policy_id='low_level_policy')
            observation, reward, f_done, info = env.step(action)
            robotPos = env.flat_env.parts["lwaist"].get_position()
            robotPos[2] = 0
            walkTarget = np.array([env.flat_env.walk_target_x, env.flat_env.walk_target_y, 1])
            drawLine(robotPos, robotPos + env.targetHighLevel * env.targetHighLevelLen, [0, 1, 0])
            drawLine(robotPos + np.array([0, 0, 1]), robotPos + walkTarget, [0, 0, 1])
            done = f_done['__all__']
            if(done):
                print(np.rad2deg(np.arctan2(env.targetHighLevel[1], env.targetHighLevel[0])))
            # print(observation)
            # drawText(str(env.frame), env.flat_env.parts["lwaist"].get_position() + np.array([0, 0, 1]), [0, 1, 0], 1.0/30)
            # drawText(str(env.deltaJoints), env.flat_env.parts["lwaist"].get_position() + np.array([1, 0, 1]), [1, 0, 0], 1.0/30)
            # drawText(str(env.deltaEndPoints), env.flat_env.parts["lwaist"].get_position() + np.array([-1, 0, 1]), [0, 0, 1], 1.0/30)
            # TODO: visualisasikan frame yang sebenarnya

            time.sleep(1.0 / fps)

            keys = pybullet.getKeyboardEvents()
            if qKey in keys and keys[qKey] & pybullet.KEY_WAS_TRIGGERED:
                print("QUIT")
                doneAll = True
            elif rKey in keys and keys[rKey] & pybullet.KEY_WAS_TRIGGERED:
                done = True
    env.close()
    ray.shutdown()