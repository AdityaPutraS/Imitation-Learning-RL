from hier_env import HierarchicalHumanoidEnv
from low_level_env import LowLevelHumanoidEnv
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune import function
from custom_callback import RewardLogCallback
from ray.rllib.agents.ddpg.apex import APEX_DDPG_DEFAULT_CONFIG
import numpy as np
from humanoid import CustomHumanoidRobot

def make_env_low(env_config):
    import pybullet_envs
    return LowLevelHumanoidEnv(reference_name="motion09_03", useCustomEnv=False, customRobot=CustomHumanoidRobot())


def make_env_hier(env_config):
    import pybullet_envs
    return HierarchicalHumanoidEnv()


def policy_mapping_fn(agent_id):
    if agent_id.startswith("low_level_"):
        return "low_level_policy"
    else:
        return "high_level_policy"

ENV_LOW = "HumanoidBulletEnv-v0-Low"
config_low = {
    "env": ENV_LOW,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 10,
    "log_level": "WARN",
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": 0.995,
    "lambda": 0.95,
    "clip_param": 0.2,
    "kl_coeff": 1.0,
    "num_sgd_iter": 20,
    "lr": 0.00005,
    "vf_clip_param": 10,
    "sgd_minibatch_size": 512,
    "train_batch_size": 6000,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "framework": "tf",
}

config_low_search = {
    "env": ENV_LOW,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 10,
    "log_level": "WARN",
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": tune.quniform(0.8, 0.9997, 0.0001),
    "lambda": tune.quniform(0.9, 1, 0.01),
    "clip_param": tune.quniform(0.1, 0.3, 0.1),
    "kl_coeff": tune.quniform(0.3, 1, 0.1),
    "lr": 5e-5,
    "vf_clip_param": 10,
    "num_sgd_iter": tune.choice([10, 20, 30]),
    "sgd_minibatch_size": 4096,
    "train_batch_size": tune.randint(8192, 40000),
    "vf_loss_coeff": tune.quniform(0.5, 1, 0.01),
    "entropy_coeff": tune.quniform(0, 0.01, 0.0001),
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "framework": "tf",
}

config_low_best = {
    "env": ENV_LOW,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 5,
    "log_level": "WARN",
    "num_gpus": 1,
    "monitor": False,
    "evaluation_num_episodes": 50,
    "gamma": 0.99,
    "lambda": 0.9,
    "clip_param": 0.5,
    "kl_coeff": 0.2,
    "num_sgd_iter": 30,
    "lr": 5e-5,
    "vf_clip_param": 10,
    "sgd_minibatch_size": 4096,
    "train_batch_size": 8192,
    "vf_loss_coeff": 1,
    "entropy_coeff": 0,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "framework": "tf",
}

config_low_best_2 = {
    "env": ENV_LOW,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 20,
    "log_level": "WARN",
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": 0.9997,
    "lambda": 0.9,
    "clip_param": 0.01,
    "kl_coeff": 1.0,
    "num_sgd_iter": 1,
    "lr": 0.001,
    "vf_clip_param": 10,
    "sgd_minibatch_size": 4096,
    "train_batch_size": 8192,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "framework": "tf",
}

config_low_pg = {
    "env": ENV_LOW,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 20,
    "log_level": "WARN",
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": 0.995,
    "lr": 0.005,
    "train_batch_size": 300,
    "model": {
        "fcnet_hiddens": [256, 128],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "rollout_fragment_length": 200,
}

config_low_apex_ddpg = APEX_DDPG_DEFAULT_CONFIG.copy()
config_low_apex_ddpg.update({
    "env": ENV_LOW,
    "callbacks": RewardLogCallback,
    "num_workers": 3,
    "num_envs_per_worker": 10,
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": 0.995,
    "critic_lr": 0.005,
    "actor_lr": 0.005,
    "train_batch_size": 6000,
    "actor_hiddens": [256, 128],
    "actor_hidden_activation": "tanh",
    "critic_hiddens": [256, 128],
    "critic_hidden_activation": "tanh",
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "timesteps_per_iteration": 24000,
    "min_iter_time_s": 10,
})
config_low_ddpg = {
    "env": ENV_LOW,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 20,
    "log_level": "WARN",
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": 0.995,
    "critic_lr": 0.005,
    "actor_lr": 0.005,
    "train_batch_size": 6000,
    "actor_hiddens": [256, 128],
    "actor_hidden_activation": "tanh",
    "critic_hiddens": [256, 128],
    "critic_hidden_activation": "tanh",
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "timesteps_per_iteration": 24000,
}

config_low_impala = {
    "env": ENV_LOW,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 20,
    "log_level": "WARN",
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": 0.995,
    "lr": 0.0005,
    "num_sgd_iter": 20,
    "train_batch_size": 6000,
    "model": {
        "fcnet_hiddens": [256, 128],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
}

config_low_a2c = {
    "env": ENV_LOW,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 20,
    "log_level": "WARN",
    "num_gpus": 1,
    "monitor": True,
    "evaluation_num_episodes": 50,
    "gamma": 0.995,
    "lr": 0.005,
    "vf_loss_coeff": 1.0,
    "entropy_coeff": 0.0,
    "train_batch_size": 6000,
    "model": {
        "fcnet_hiddens": [256, 128],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "rollout_fragment_length": 10,
}

single_env = HierarchicalHumanoidEnv()
ENV_HIER = "HumanoidBulletEnvHier-v0"
highLevelPolicy = (
    None,
    single_env.high_level_obs_space,
    single_env.high_level_act_space,
    {
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
            "free_log_std": True,
        },
    },
)

lowLevelPolicy = (
    None,
    single_env.low_level_obs_space,
    single_env.low_level_act_space,
    {
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
            "free_log_std": True,
        },
    },
)

config_hier = {
    "env": ENV_HIER,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 20,
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
    "lr": 0.00005,
    "vf_clip_param": 10,
    "sgd_minibatch_size": 512,
    "train_batch_size": 6000,
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
}

config_hier_best = {
    "env": ENV_HIER,
    "callbacks": RewardLogCallback,
    "num_workers": 6,
    "num_envs_per_worker": 10,
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
    "gamma": 0.9997,
    "lambda": 0.901324,
    "clip_param": 0.5,
    "kl_coeff": 0.505869,
    "num_sgd_iter": 30,
    "lr": 5.927e-5,
    "vf_clip_param": 10,
    "sgd_minibatch_size": 4096,
    "train_batch_size": 8431,
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
}

register_env(ENV_HIER, make_env_hier)
register_env(ENV_LOW, make_env_low)