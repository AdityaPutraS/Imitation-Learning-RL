from hier_env import HierarchicalHumanoidEnv
from low_level_env import LowLevelHumanoidEnv
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune import function
from custom_callback import RewardLogCallback

def make_env_low(env_config):
    import pybullet_envs
    return LowLevelHumanoidEnv()


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
    "num_envs_per_worker": 20,
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
    "vf_clip_param": 10,
    "sgd_minibatch_size": 12000,
    "train_batch_size": 36000,
    "model": {
        "fcnet_hiddens": [1024, 512],
        "fcnet_activation": "tanh",
        "free_log_std": True,
    },
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
    "framework": "tf",
}

single_env = HierarchicalHumanoidEnv()
ENV_HIER = "HumanoidBulletEnvHier-v0"
highLevelPolicy = (
    None,
    single_env.high_level_obs_space,
    single_env.high_level_act_space,
    {
        "model": {
            "fcnet_hiddens": [512, 256],
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
            "fcnet_hiddens": [1024, 512],
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
    "lr": 0.0005,
    "vf_clip_param": 10,
    "sgd_minibatch_size": 12000,
    "train_batch_size": 36000,
    "batch_mode": "complete_episodes",
    "observation_filter": "NoFilter",
}

register_env(ENV_HIER, make_env_hier)
register_env(ENV_LOW, make_env_low)