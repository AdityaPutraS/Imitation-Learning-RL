import gym
import pybullet
import pybullet_envs
import pybullet_data
from pybullet_envs.robot_bases import BodyPart
import os

from gym.spaces import Box, Discrete, Tuple
import logging
import random

import pandas as pd
import numpy as np

from ray.rllib.env import MultiAgentEnv

from low_level_env import LowLevelHumanoidEnv

from scipy.spatial.transform import Rotation as R
from math_util import rotFrom2Vec

logger = logging.getLogger(__name__)


def getJointPos(df, joint, multiplier=1):
    x = df[joint + "_Xposition"]
    y = df[joint + "_Yposition"]
    z = df[joint + "_Zposition"]
    return np.array([x, y, z]) * multiplier

class HierarchicalHumanoidEnv(MultiAgentEnv):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        self.flat_env = gym.make("HumanoidBulletEnv-v0")
        # self.action_space = self.flat_env.action_space
        # self.observation_space = self.flat_env.observation_space

        reference_list = [
            ("Joints CSV/walk8_1JointPos.csv", "Joints CSV/walk8_1JointVecFromHip.csv"),
        ]
        num_reference = len(reference_list)

        self.high_level_obs_space = self.flat_env.observation_space
        self.high_level_act_space = Box(
            low=-1.0, high=1.0, shape=[2]
        )  # [target_x, target_y]

        self.low_level_obs_space = Box(
            low=-np.inf, high=np.inf, shape=[1 + 5 + 17*2 + 2]
        )
        self.low_level_act_space = self.flat_env.action_space

        self.step_per_level = 100
        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0

        # self.joints_df = [pd.read_csv(path[0]) for path in reference_list]
        # self.end_point_df = [pd.read_csv(path[1]) for path in reference_list]
        self.joints_df = pd.read_csv(reference_list[0][0])
        self.end_point_df = pd.read_csv(reference_list[0][1])

        self.frame = 0
        self.frame_update_cnt = 0
        self.max_frame = len(self.joints_df)

        self.max_delta_j_rad = np.deg2rad(5)

        self.rng = np.random.default_rng()

        self.joint_map = {
            "right_knee": "rightKnee",
            "right_hip_x": "rightHipX",
            "right_hip_y": "rightHipY",
            "right_hip_z": "rightHipZ",
            "left_knee": "leftKnee",
            "left_hip_x": "leftHipX",
            "left_hip_y": "leftHipY",
            "left_hip_z": "leftHipZ",
        }

        self.joint_weight = {
            "right_knee": 3,
            "right_hip_x": 1,
            "right_hip_y": 3,
            "right_hip_z": 1,
            "left_knee": 3,
            "left_hip_x": 1,
            "left_hip_y": 3,
            "left_hip_z": 1,
        }
        self.joint_weight_sum = sum(self.joint_weight.values())

        self.end_point_map = {
            "link0_11": "RightLeg",
            "right_foot": "RightFoot",
            "link0_18": "LeftLeg",
            "left_foot": "LeftFoot",
        }

        self.end_point_weight = {
            "link0_11": 2,
            "right_foot": 1,
            "link0_18": 2,
            "left_foot": 1,
        }
        self.end_point_weight_sum = sum(self.end_point_weight.values())

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0
        self.lowTargetScore = 0
        self.highTargetScore = 0

        self.target = np.array([1, 0, 0])
        self.targetHighLevel = np.array([0, 0, 0])
        self.skipFrame = 1

    def close(self):
        self.flat_env.close()

    def render(self, mode="human"):
        return self.flat_env.render(mode)

    def setJointsOrientation(self, idx):
        jointsRef = self.joints_df.iloc[idx]
        self.flat_env.jdict["right_knee"].set_state(jointsRef["rightKnee"], 0)

        self.flat_env.jdict["right_hip_x"].set_state(jointsRef["rightHipX"], 0)
        self.flat_env.jdict["right_hip_y"].set_state(jointsRef["rightHipY"], 0)
        self.flat_env.jdict["right_hip_z"].set_state(jointsRef["rightHipZ"], 0)

        self.flat_env.jdict["left_knee"].set_state(jointsRef["leftKnee"], 0)

        self.flat_env.jdict["left_hip_x"].set_state(jointsRef["leftHipX"], 0)
        self.flat_env.jdict["left_hip_y"].set_state(jointsRef["leftHipY"], 0)
        self.flat_env.jdict["left_hip_z"].set_state(jointsRef["leftHipZ"], 0)

    def incFrame(self, inc):
        self.frame += inc
        self.frame = self.frame % (self.max_frame - 1)
    
    def reset(self):
        # Insialisasi dengan posisi awal random sesuai referensi
        return self.resetFromFrame(self.rng.integers(0, self.max_frame - 5), self.rng.integers(-15, 15))

    def resetFromFrame(self, frame, resetYaw=0):
        self.flat_env.reset()

        randomX = self.rng.integers(-20, 20)
        randomY = self.rng.integers(-20, 20)
        self.target = np.array([randomX, randomY, 0])
        self.targetHighLevel = np.array([0, 0, 0])
        self.flat_env.walk_target_x = randomX
        self.flat_env.walk_target_y = randomY

        randomProgress = self.rng.random() * 0.8
        self.flat_env.parts["torso"].reset_position([randomX * randomProgress, randomY * randomProgress, 1.15])
        self.flat_env.parts["torso"].reset_orientation(R.from_euler('z', resetYaw, degrees=True).as_quat())

        self.frame = frame
        self.frame_update_cnt = 0
        self.setJointsOrientation(self.frame)

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0
        self.lowTargetScore = 0
        self.highTargetScore = 0

        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0
        
        self.skipFrame = 2

        self.incFrame(self.skipFrame)

        # self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        self.low_level_agent_id = "low_level_agent"

        self.cur_obs = self.flat_env.robot.calc_state()
        return {
            "high_level_agent": self.cur_obs
        }

    def resetNonFrame(self):
        self.cur_obs = self.flat_env.reset()
        # self.flat_env.parts['torso'].reset_position([0, 0, 1.15])

        self.frame = 0
        self.frame_update_cnt = 0

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0
        self.lowTargetScore = 0
        self.highTargetScore = 0

        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0

        self.target = np.array([1, 0, 0])
        self.targetHighLevel = np.array([0, 0, 0])
        
        self.skipFrame = 1

        # self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        self.low_level_agent_id = "low_level_agent"
        return {
            "high_level_agent": self.cur_obs,
        }

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "high_level_agent" in action_dict:
            return self.high_level_step(action_dict["high_level_agent"])
        else:
            return self.low_level_step(list(action_dict.values())[0])

    def calcJointScore(self):
        deltaJoints = 0
        jointsRef = self.joints_df.iloc[self.frame]

        for jMap in self.joint_map:
            deltaJoints += (
                np.abs(
                    self.flat_env.jdict[jMap].get_position()
                    - jointsRef[self.joint_map[jMap]]
                )
                * self.joint_weight[jMap]
            )

        return np.exp(-5 * deltaJoints / self.joint_weight_sum)

    def calcEndPointScore(self):
        deltaEndPoint = 0
        endPointRef = self.end_point_df.iloc[self.frame]

        base_pos = self.flat_env.parts["lwaist"].get_position()
        r = R.from_euler('z', np.arctan2(self.targetHighLevel[1], self.targetHighLevel[0]))

        for epMap in self.end_point_map:
            v1 = self.flat_env.parts[epMap].get_position()
            v2 = base_pos + r.apply(getJointPos(endPointRef, self.end_point_map[epMap]))
            deltaVec = v2 - v1
            deltaEndPoint += np.linalg.norm(deltaVec) * self.end_point_weight[epMap]

        return 2 * np.exp(-10 * deltaEndPoint / self.end_point_weight_sum)
    
    def calcHighLevelTargetScore(self):
        vTarget = self.target
        vRobot = self.flat_env.parts['torso'].get_position()

        # Map ke 2d XY
        vTarget[2] = 0
        vRobot[2] = 0

        # Hitung jarak
        dRobotTarget = np.linalg.norm(vTarget - vRobot)

        return 2 * np.exp(-dRobotTarget/5)
    
    def calcLowLevelTargetScore(self):
        vTargetHL = self.targetHighLevel
        vRobot = self.flat_env.parts['torso'].get_position()

        vTargetHL[2] = 0
        vRobot[2] = 0

        # Hitung jarak
        dRobotTargetHL = np.linalg.norm(vTargetHL - vRobot)

        return 2 * np.exp(-dRobotTargetHL)

    def calcJumpReward(self, obs):
        return 0

    def getLowLevelObs(self):
        # Berikan low level agent nilai joint dan target high level
        deltaZ = self.cur_obs[0]
        robotInfo = self.cur_obs[3:3+5]
        jointVal = self.cur_obs[8:8+17*2]
        
        vTargetHL = self.targetHighLevel
        vRobot = self.flat_env.parts['torso'].get_position()

        return np.hstack((deltaZ, robotInfo, jointVal, vTargetHL[0] - vRobot[0], vTargetHL[1] - vRobot[1]))

    def getHighLevelObs(self):
        return self.cur_obs

    def high_level_step(self, action):
        self.current_goal = action

        self.targetHighLevel = np.array([action[0], action[1], 0])
        
        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps += 1
        # self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)

        obs = {self.low_level_agent_id: self.getLowLevelObs()}
        rew = {self.low_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def low_level_step(self, action):
        self.steps_remaining_at_level -= 1

        # Step in the actual env
        f_obs, f_rew, f_done, _ = self.flat_env.step(action)

        self.cur_obs = f_obs

        # Calculate reward
        self.deltaJoints = self.calcJointScore()  # Rentang 0 - 1
        self.deltaEndPoints = self.calcEndPointScore()  # Rentang 0 - 1
        self.baseReward = f_rew
        self.lowTargetScore = self.calcLowLevelTargetScore()
        self.highTargetScore = self.calcHighLevelTargetScore()

        jumpReward = self.calcJumpReward(f_obs)  # Untuk task lompat

        reward = [self.baseReward, self.deltaJoints, self.deltaEndPoints, self.lowTargetScore, jumpReward]
        rewardWeight = [1, 0.1, 1.5, 4, 0]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        if self.deltaJoints >= 0.2 and self.deltaEndPoints >= 0.15:
            self.incFrame(self.skipFrame)
            self.frame_update_cnt = 0
        else:
            self.frame_update_cnt += 1
            if self.frame_update_cnt > 10:
                self.incFrame(self.skipFrame)
                self.frame_update_cnt = 0

        rew, obs = dict(), dict()
        # Handle env termination & transitions back to higher level
        done = {"__all__": False}
        if f_done:
            done["__all__"] = True
            rew["high_level_agent"] = (self.highTargetScore + self.lowTargetScore / (self.steps_remaining_at_level + 1))
            obs["high_level_agent"] = self.getHighLevelObs()
            obs[self.low_level_agent_id] = self.getLowLevelObs()
            rew[self.low_level_agent_id] = totalReward
        elif self.steps_remaining_at_level <= 0:
            # done[self.low_level_agent_id] = True
            rew["high_level_agent"] = (self.highTargetScore + self.lowTargetScore)
            obs["high_level_agent"] = self.getHighLevelObs()
        else:
            obs = {self.low_level_agent_id: self.getLowLevelObs()}
            rew = {self.low_level_agent_id: totalReward}

        return obs, rew, done, {}