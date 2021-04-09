import numpy as np

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from typing import List, Dict, Callable, Any, TYPE_CHECKING

class RewardLogCallback(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):

        episode.user_data["delta_joints"] = []
        episode.user_data["delta_end_point"] = []
        episode.user_data["base_reward"] = []

        episode.user_data["low_target_score"] = []
        episode.user_data["high_target_score"] = []

        episode.user_data["delta_joints_velocity"] = []
        episode.user_data["alive_reward"] = []

        episode.user_data["dist_from_origin"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        dj = base_env.get_unwrapped()[0].deltaJoints
        ep = base_env.get_unwrapped()[0].deltaEndPoints
        br = base_env.get_unwrapped()[0].baseReward
        
        episode.user_data["delta_joints"].append(dj)
        episode.user_data["delta_end_point"].append(ep)
        episode.user_data["base_reward"].append(br)

        lts = base_env.get_unwrapped()[0].lowTargetScore
        hts = base_env.get_unwrapped()[0].highTargetScore
        episode.user_data["low_target_score"].append(lts)
        episode.user_data["high_target_score"].append(hts)

        djv = base_env.get_unwrapped()[0].deltaVelJoints
        ar = base_env.get_unwrapped()[0].aliveReward
        episode.user_data["delta_joints_velocity"].append(djv)
        episode.user_data["alive_reward"].append(ar)

        robotDist = np.linalg.norm(base_env.get_unwrapped()[0].starting_ep_pos)
        episode.user_data["dist_from_origin"].append(robotDist)
        
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        mean_dj = np.mean(episode.user_data["delta_joints"])
        mean_ep = np.mean(episode.user_data["delta_end_point"])
        mean_br = np.mean(episode.user_data["base_reward"])

        mean_lts = np.mean(episode.user_data["low_target_score"])
        mean_hts = np.mean(episode.user_data["high_target_score"])

        mean_djv = np.mean(episode.user_data["delta_joints_velocity"])
        mean_ar = np.mean(episode.user_data["alive_reward"])

        mean_dfo = np.mean(episode.user_data["dist_from_origin"])

        episode.custom_metrics["delta_joints"] = mean_dj
        episode.custom_metrics["delta_end_point"] = mean_ep
        episode.custom_metrics["base_reward"] = mean_br
        
        episode.custom_metrics["low_target_score"] = mean_lts
        episode.custom_metrics["high_target_score"] = mean_hts

        episode.custom_metrics["delta_joints_velocity"] = mean_djv
        episode.custom_metrics["alive_reward"] = mean_ar

        episode.custom_metrics["dist_from_origin"] = mean_dfo