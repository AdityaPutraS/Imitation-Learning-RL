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
        episode.user_data["delta_joints_velocity"] = []
        episode.user_data["base_reward"] = []

        episode.user_data["low_target_score"] = []

        episode.user_data["high_target_score"] = []
        episode.user_data["drift_score"] = []

        episode.user_data["alive_reward"] = []

        episode.user_data["dist_from_origin"] = []

        episode.user_data["electricity_score"] = []
        episode.user_data["joint_limit_score"] = []

        episode.user_data["body_posture_score"] = []

        # episode.user_data["delta_joints_low"] = []
        # episode.user_data["delta_joints_velocity_low"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        dj = base_env.get_unwrapped()[0].deltaJoints
        ep = base_env.get_unwrapped()[0].deltaEndPoints
        lts = base_env.get_unwrapped()[0].lowTargetScore
        djv = base_env.get_unwrapped()[0].deltaVelJoints
        bps = base_env.get_unwrapped()[0].bodyPostureScore

        hts = base_env.get_unwrapped()[0].highTargetScore
        ds = base_env.get_unwrapped()[0].driftScore

        br = base_env.get_unwrapped()[0].baseReward
        ar = base_env.get_unwrapped()[0].aliveReward

        es = base_env.get_unwrapped()[0].electricityScore
        jls = base_env.get_unwrapped()[0].jointLimitScore

        # dj_low = base_env.get_unwrapped()[0].deltaJoints_low
        # djv_low = base_env.get_unwrapped()[0].deltaVelJoints_low

        episode.user_data["delta_joints"].append(dj)
        episode.user_data["delta_end_point"].append(ep)
        episode.user_data["low_target_score"].append(lts)
        episode.user_data["delta_joints_velocity"].append(djv)
        episode.user_data["body_posture_score"].append(bps)

        episode.user_data["high_target_score"].append(hts)
        episode.user_data["drift_score"].append(ds)
        
        episode.user_data["base_reward"].append(br)
        episode.user_data["alive_reward"].append(ar)

        episode.user_data["electricity_score"].append(es)
        episode.user_data["joint_limit_score"].append(jls)

        # episode.user_data["delta_joints_low"].append(dj_low)
        # episode.user_data["delta_joints_velocity_low"].append(djv_low)

        # robotDist = np.linalg.norm(base_env.get_unwrapped()[0].starting_ep_pos)
        robotDist = np.linalg.norm(base_env.get_unwrapped()[0].robot_pos)
        episode.user_data["dist_from_origin"].append(robotDist)
        
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        mean_dj = np.mean(episode.user_data["delta_joints"])
        mean_ep = np.mean(episode.user_data["delta_end_point"])
        mean_br = np.mean(episode.user_data["base_reward"])

        mean_lts = np.mean(episode.user_data["low_target_score"])

        mean_hts = np.mean(episode.user_data["high_target_score"])
        mean_ds = np.mean(episode.user_data["drift_score"])

        mean_djv = np.mean(episode.user_data["delta_joints_velocity"])
        mean_ar = np.mean(episode.user_data["alive_reward"])

        mean_dfo = np.mean(episode.user_data["dist_from_origin"])

        mean_es = np.mean(episode.user_data["electricity_score"])
        mean_jls = np.mean(episode.user_data["joint_limit_score"])

        mean_bps = np.mean(episode.user_data["body_posture_score"])

        # mean_dj_low = np.mean(episode.user_data["delta_joints_low"])
        # mean_djv_low = np.mean(episode.user_data["delta_joints_velocity_low"])

        episode.custom_metrics["delta_joints"] = mean_dj
        episode.custom_metrics["delta_end_point"] = mean_ep
        episode.custom_metrics["base_reward"] = mean_br
        
        episode.custom_metrics["low_target_score"] = mean_lts

        episode.custom_metrics["high_target_score"] = mean_hts
        episode.custom_metrics["drift_score"] = mean_ds

        episode.custom_metrics["delta_joints_velocity"] = mean_djv
        episode.custom_metrics["alive_reward"] = mean_ar

        episode.custom_metrics["dist_from_origin"] = mean_dfo

        episode.custom_metrics["electricity_score"] = mean_es
        episode.custom_metrics["joint_limit_score"] = mean_jls

        episode.custom_metrics["body_posture_score"] = mean_bps

        # episode.custom_metrics["delta_joints_low"] = mean_dj_low
        # episode.custom_metrics["delta_joints_velocity_low"] = mean_djv_low