import gym
import pybullet
import pybullet_envs
import pybullet_data
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv
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
from math_util import rotFrom2Vec, map_seq

logger = logging.getLogger(__name__)


def getJointPos(df, joint, multiplier=1):
    x = df[joint + "_Xposition"]
    y = df[joint + "_Yposition"]
    z = df[joint + "_Zposition"]
    return np.array([x, y, z]) * multiplier


def drawLine(c1, c2, color):
    return pybullet.addUserDebugLine(c1, c2, lineColorRGB=color, lineWidth=5, lifeTime=0.1)


class HierarchicalHumanoidEnv(MultiAgentEnv):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        self.flat_env = HumanoidBulletEnv()
        # self.action_space = self.flat_env.action_space
        # self.observation_space = self.flat_env.observation_space

        self.motion_list = ["motion08_03", "motion09_03"]

        self.high_level_obs_space = self.flat_env.observation_space
        self.high_level_act_space = Box(
            low=-1, high=1, shape=[4]
        )  # cos(target), sin(target), mapping selected motion

        self.low_level_obs_space = Box(low=-np.inf, high=np.inf, shape=[44 + 8 * 2])
        self.low_level_act_space = self.flat_env.action_space

        self.step_per_level = 30
        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0

        self.joints_df = [
            pd.read_csv("Processed Relative Joints CSV/{}JointPosRad.csv".format(mot))
            for mot in self.motion_list
        ]

        self.joints_vel_df = [
            pd.read_csv(
                "Processed Relative Joints CSV/{}JointSpeedRadSec.csv".format(mot)
            )
            for mot in self.motion_list
        ]

        self.joints_rel_df = [
            pd.read_csv(
                "Processed Relative Joints CSV/{}JointPosRadRelative.csv".format(mot)
            )
            for mot in self.motion_list
        ]

        self.end_point_df = [
            pd.read_csv(
                "Processed Relative Joints CSV/{}JointVecFromHip.csv".format(mot)
            )
            for mot in self.motion_list
        ]

        self.cur_timestep = 0
        self.max_timestep = 3000

        self.frame = 0
        self.frame_update_cnt = 0

        self.max_frame = [len(df)-1 for df in self.joints_df]

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

        self.joint_vel_weight = {
            "right_knee": 1,
            "right_hip_x": 1,
            "right_hip_y": 1,
            "right_hip_z": 1,
            "left_knee": 1,
            "left_hip_x": 1,
            "left_hip_y": 1,
            "left_hip_z": 1,
        }
        self.joint_vel_weight_sum = sum(self.joint_vel_weight.values())

        self.end_point_map = {
            "link0_11": "RightLeg",
            "right_foot": "RightFoot",
            "link0_18": "LeftLeg",
            "left_foot": "LeftFoot",
        }

        self.end_point_weight = {
            "link0_11": 1,
            "right_foot": 1.1,
            "link0_18": 1,
            "left_foot": 1.1,
        }
        self.end_point_weight_sum = sum(self.end_point_weight.values())

        self.initReward()

        self.target = np.array([1, 0, 0])
        self.targetHighLevel = np.array([0, 0, 0])
        self.skipFrame = 1

        # Menandakan motion apa yang harus dijalankan sekarang dan pada frame berapa
        self.selected_motion = 0
        self.selected_motion_frame = 0

        self.targetHighLevelLen = 20
        self.starting_ep_pos = np.array([0, 0, 0])

    def initReward(self):
        self.deltaJoints = 0
        self.deltaVelJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0
        self.lowTargetScore = 0
        self.highTargetScore = 0
        self.aliveReward = 0

    def close(self):
        self.flat_env.close()

    def render(self, mode="human"):
        return self.flat_env.render(mode)

    def setJointsOrientation(self, df_idx, frame_idx):
        jointsRef = self.joints_df[df_idx].iloc[frame_idx]
        jointsVelRef = self.joints_vel_df[df_idx].iloc[frame_idx]

        self.flat_env.jdict["abdomen_x"].set_state(0, 0)
        self.flat_env.jdict["abdomen_y"].set_state(0, 0)
        self.flat_env.jdict["abdomen_z"].set_state(0, 0)

        self.flat_env.jdict["right_knee"].set_state(
            jointsRef["rightKnee"], jointsVelRef["rightKnee"]
        )

        self.flat_env.jdict["right_hip_x"].set_state(
            jointsRef["rightHipX"], jointsVelRef["rightHipX"]
        )
        self.flat_env.jdict["right_hip_y"].set_state(
            jointsRef["rightHipY"], jointsVelRef["rightHipY"]
        )
        self.flat_env.jdict["right_hip_z"].set_state(
            jointsRef["rightHipZ"], jointsVelRef["rightHipZ"]
        )

        self.flat_env.jdict["left_knee"].set_state(
            jointsRef["leftKnee"], jointsVelRef["leftKnee"]
        )

        self.flat_env.jdict["left_hip_x"].set_state(
            jointsRef["leftHipX"], jointsVelRef["leftHipX"]
        )
        self.flat_env.jdict["left_hip_y"].set_state(
            jointsRef["leftHipY"], jointsVelRef["leftHipY"]
        )
        self.flat_env.jdict["left_hip_z"].set_state(
            jointsRef["leftHipZ"], jointsVelRef["leftHipZ"]
        )

    def incFrame(self, inc):
        # self.frame = (self.frame + inc) % (self.max_frame - 1)
        self.selected_motion_frame = (self.selected_motion_frame + inc) % (
            self.max_frame[self.selected_motion] - 1
        )

    def reset(self):
        # Insialisasi dengan posisi awal random sesuai referensi
        return self.resetWithYaw(self.rng.integers(-10, 10))

    def resetWithYaw(self, resetYaw):
        self.flat_env.reset()

        self.cur_timestep = 0

        # randomX = self.rng.integers(-500, 50)
        # randomY = self.rng.integers(-500, 500)
        randomX = 1e3
        randomY = 0
        self.target = np.array([randomX, randomY, 0])

        # randomProgress = self.rng.random() * 0.8
        randomProgress = 0
        self.flat_env.parts["torso"].reset_position(
            [randomX * randomProgress, randomY * randomProgress, 1.15]
        )
        self.flat_env.parts["torso"].reset_orientation(
            R.from_euler(
                "z", np.rad2deg(np.arctan2(randomY, randomX)) + resetYaw, degrees=True
            ).as_quat()
        )
        self.starting_ep_pos = np.array([randomX * randomProgress, randomY * randomProgress, 0])
        self.targetHighLevel = (
            np.array([np.cos(np.deg2rad(-resetYaw)), np.sin(np.deg2rad(-resetYaw)), 0])
            * self.targetHighLevelLen
        )

        self.flat_env.walk_target_x = self.targetHighLevel[0] * 50
        self.flat_env.walk_target_y = self.targetHighLevel[1] * 50

        self.selected_motion = 0
        self.selected_motion_frame = 0

        self.setJointsOrientation(self.selected_motion, self.selected_motion_frame)

        self.initReward()

        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0

        self.incFrame(self.skipFrame)

        # self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        self.low_level_agent_id = "low_level_agent"

        self.cur_obs = self.flat_env.robot.calc_state()
        return {"high_level_agent": self.getHighLevelObs()}

    # def resetNonFrame(self):
    #     self.cur_obs = self.flat_env.reset()
    #     # self.flat_env.parts['torso'].reset_position([0, 0, 1.15])

    #     self.frame = 0
    #     self.frame_update_cnt = 0

    #     self.deltaJoints = 0
    #     self.deltaEndPoints = 0
    #     self.baseReward = 0
    #     self.lowTargetScore = 0
    #     self.highTargetScore = 0

    #     self.steps_remaining_at_level = self.step_per_level
    #     self.num_high_level_steps = 0

    #     self.target = np.array([1, 0, 0])
    #     self.targetHighLevel = np.array([0, 0, 0])

    #     self.skipFrame = 1

    #     # self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
    #     self.low_level_agent_id = "low_level_agent"
    #     return {
    #         "high_level_agent": self.cur_obs,
    #     }

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "high_level_agent" in action_dict:
            return self.high_level_step(action_dict["high_level_agent"])
        else:
            return self.low_level_step(list(action_dict.values())[0])

    def calcJointScore(self):
        deltaJoints = 0
        jointsRef = self.joints_df[self.selected_motion].iloc[
            self.selected_motion_frame
        ]

        for jMap in self.joint_map:
            deltaJoints += (
                np.abs(
                    self.flat_env.jdict[jMap].get_position()
                    - jointsRef[self.joint_map[jMap]]
                )
                * self.joint_weight[jMap]
            )

        score = np.exp(-1 * deltaJoints / self.joint_weight_sum)
        return (score * 3) - 1.8

    def calcJointVelScore(self):
        deltaVel = 0
        jointsVelRef = self.joints_vel_df[self.selected_motion].iloc[
            self.selected_motion_frame
        ]

        for jMap in self.joint_map:
            deltaVel += (
                np.abs(
                    self.flat_env.jdict[jMap].get_velocity()
                    - jointsVelRef[self.joint_map[jMap]]
                )
                * self.joint_vel_weight[jMap]
            )

        score = np.exp(-1 * deltaVel / self.joint_vel_weight_sum)
        return (score * 3) - 1.8

    def calcEndPointScore(self):
        deltaEndPoint = 0
        endPointRef = self.end_point_df[self.selected_motion].iloc[
            self.selected_motion_frame
        ]

        # base_pos = self.flat_env.parts["lwaist"].get_position()
        r = R.from_euler(
            "z", np.arctan2(self.targetHighLevel[1], self.targetHighLevel[0])
        )

        for epMap in self.end_point_map:
            v1 = self.flat_env.parts[epMap].get_position()
            # v2 = base_pos + r.apply(getJointPos(endPointRef, self.end_point_map[epMap]))
            v2 = self.starting_ep_pos + r.apply(getJointPos(endPointRef, self.end_point_map[epMap]))
            # drawLine(v1, v2, [1, 0, 0])
            deltaVec = v2 - v1
            deltaEndPoint += np.linalg.norm(deltaVec) * self.end_point_weight[epMap]

        score = np.exp(-2 * deltaEndPoint / self.end_point_weight_sum)
        return score * 3 - 1.6
        # return 15 * np.exp(-30 * deltaEndPoint / self.end_point_weight_sum)
        # return deltaEndPoint / self.end_point_weight_sum

    def calcHighLevelTargetScore(self):
        vTarget = self.target
        vRobot = self.flat_env.parts["torso"].get_position()

        # Map ke 2d XY
        vTarget[2] = 0
        vRobot[2] = 0

        # Hitung jarak
        dRobotTarget = np.linalg.norm(vTarget - vRobot)

        return 2 * np.exp(-dRobotTarget / 5)

    # def calcLowLevelTargetScore(self):
    #     vTargetHL = self.targetHighLevel
    #     vRobot = self.flat_env.parts["torso"].get_position()

    #     vTargetHL[2] = 0
    #     vRobot[2] = 0

    #     # Hitung jarak
    #     dRobotTargetHL = np.linalg.norm(vTargetHL - vRobot)

    #     return 2 * np.exp(-dRobotTargetHL)

    def calcJumpReward(self, obs):
        return 0

    def calcAliveReward(self):
        # Didapat dari perhitungan reward alive env humanoid
        z = self.cur_obs[0] + self.flat_env.robot.initial_z
        # return +2 if z > 0.78 else -1
        return +2 if z > 0.6 else -1

    def getLowLevelObs(self):
        # [cur obs + target pose]
        jointTargetObs = []
        jointsRelRef = self.joints_rel_df[self.selected_motion].iloc[
            self.selected_motion_frame
        ]
        jointsVelRef = self.joints_vel_df[self.selected_motion].iloc[
            self.selected_motion_frame
        ]
        for jMap in self.joint_map:
            jointTargetObs.append(jointsRelRef[self.joint_map[jMap]])
            jointTargetObs.append(jointsVelRef[self.joint_map[jMap]])
        jointTargetObs = np.array(jointTargetObs)

        return np.hstack((self.cur_obs, jointTargetObs))

    def getHighLevelObs(self):
        robotPos = self.flat_env.parts["torso"].get_position()
        targetTheta = np.arctan2(
            self.target[1] - robotPos[1], self.target[0] - robotPos[0]
        )
        r, p, yaw = self.flat_env.robot.robot_body.pose().rpy()
        angleToTarget = targetTheta - yaw
        degTarget = [np.cos(angleToTarget), np.sin(angleToTarget)]

        return np.hstack((self.cur_obs[:1], degTarget, self.cur_obs[3:]))

    def high_level_step(self, action):
        # Map sudut agar berada di sekitar -15 s/d 15 derajat
        cosTarget = map_seq(action[0], -1, 1, 0, 0.9659) # 0.9659 = cos(+-15 derajat)
        sinTarget = map_seq(action[1], -1, 1, -0.2588, 0.2588) # 0.2588 = sin(15 derajat), -0.2588 = sin(-15 derajat)
        self.targetHighLevel = np.array([cosTarget, sinTarget, 0]) * self.targetHighLevelLen
        self.flat_env.walk_target_x = self.targetHighLevel[0] * 50
        self.flat_env.walk_target_y = self.targetHighLevel[1] * 50

        vRobot = self.flat_env.parts["torso"].get_position()
        self.starting_ep_pos = np.array([vRobot[0], vRobot[1], 0])
        
        self.selected_motion = int(map_seq(action[2], -1.0, 1.0, 0.0, len(self.motion_list) - 1))
        self.selected_motion_frame = int(map_seq(action[3], -1.0, 1.0, 0.0, self.max_frame[self.selected_motion] - 1))
        
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
        self.deltaVelJoints = self.calcJointVelScore()
        self.deltaEndPoints = self.calcEndPointScore()  # Rentang 0 - 1
        self.baseReward = f_rew

        self.highTargetScore = self.calcHighLevelTargetScore()


        reward = [
            self.baseReward,
            self.deltaJoints,
            self.deltaVelJoints,
            self.deltaEndPoints
        ]
        rewardWeight = [0.25, 0.25, 0.25, 1]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        self.incFrame(self.skipFrame)

        rew, obs = dict(), dict()
        # Handle env termination & transitions back to higher level
        done = {"__all__": False}
        if f_done:
            done["__all__"] = True
            rew["high_level_agent"] = self.highTargetScore + 0.2 * np.exp(-self.steps_remaining_at_level / self.step_per_level)
            obs["high_level_agent"] = self.getHighLevelObs()
            obs[self.low_level_agent_id] = self.getLowLevelObs()
            rew[self.low_level_agent_id] = totalReward
        elif self.steps_remaining_at_level <= 0:
            # done[self.low_level_agent_id] = True
            rew["high_level_agent"] = self.highTargetScore + 0.2
            obs["high_level_agent"] = self.getHighLevelObs()
        else:
            obs = {self.low_level_agent_id: self.getLowLevelObs()}
            rew = {self.low_level_agent_id: totalReward}

        return obs, rew, done, {}