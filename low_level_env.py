import gym
import pybullet
import pybullet_envs
import pybullet_data
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv
from gym.spaces import Box, Discrete, Tuple
import logging
import random

import pandas as pd
import numpy as np

from ray.rllib.env import MultiAgentEnv

from scipy.spatial.transform import Rotation as R
from math_util import rotFrom2Vec

logger = logging.getLogger(__name__)


def getJointPos(df, joint, multiplier=1):
    x = df[joint + "_Xposition"]
    y = df[joint + "_Yposition"]
    z = df[joint + "_Zposition"]
    return np.array([x, y, z]) * multiplier


def drawLine(c1, c2, color):
    return pybullet.addUserDebugLine(c1, c2, lineColorRGB=color, lineWidth=5)


class LowLevelHumanoidEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        self.flat_env = HumanoidBulletEnv()

        # self.observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=[1 + 5 + 17 * 2 + 2 + 8]
        # )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=[44 + 8 * 2])
        self.action_space = self.flat_env.action_space

        reference_name = "walk09_01"

        self.joints_df = pd.read_csv(
            "Processed Relative Joints CSV/{}JointPosRad.csv".format(reference_name)
        )
        self.joints_vel_df = pd.read_csv(
            "Processed Relative Joints CSV/{}JointSpeedRadSec.csv".format(reference_name)
        )
        self.joints_rel_df = pd.read_csv(
            "Processed Relative Joints CSV/{}JointPosRadRelative.csv".format(reference_name)
        )
        self.end_point_df = pd.read_csv(
            "Processed Relative Joints CSV/{}JointVecFromHip.csv".format(reference_name)
        )

        self.cur_timestep = 0
        self.max_timestep = 2000

        self.frame = 0
        self.frame_update_cnt = 0

        # Untuk 08_03, frame 0 - 125 merupakan siklus (2 - 127 di blender)
        # Untuk 09_01, frame 0 - 90 merupakan siklus (1 - 91 di blender)
        self.max_frame = 90

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
            "right_foot": 3,
            "link0_18": 1,
            "left_foot": 3,
        }
        self.end_point_weight_sum = sum(self.end_point_weight.values())

        self.initReward()

        self.targetHighLevel = np.array([0, 0, 0])
        self.skipFrame = 1

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

    def setJointsOrientation(self, idx):
        jointsRef = self.joints_df.iloc[idx]
        jointsVelRef = self.joints_vel_df.iloc[idx]

        self.flat_env.jdict["abdomen_x"].set_state(0, 0)
        self.flat_env.jdict["abdomen_y"].set_state(0, 0)
        self.flat_env.jdict["abdomen_z"].set_state(0, 0)

        self.flat_env.jdict["right_knee"].set_state(jointsRef["rightKnee"], jointsVelRef["rightKnee"])

        self.flat_env.jdict["right_hip_x"].set_state(jointsRef["rightHipX"], jointsVelRef["rightHipX"])
        self.flat_env.jdict["right_hip_y"].set_state(jointsRef["rightHipY"], jointsVelRef["rightHipY"])
        self.flat_env.jdict["right_hip_z"].set_state(jointsRef["rightHipZ"], jointsVelRef["rightHipZ"])

        self.flat_env.jdict["left_knee"].set_state(jointsRef["leftKnee"], jointsVelRef["leftKnee"])

        self.flat_env.jdict["left_hip_x"].set_state(jointsRef["leftHipX"], jointsVelRef["leftHipX"])
        self.flat_env.jdict["left_hip_y"].set_state(jointsRef["leftHipY"], jointsVelRef["leftHipY"])
        self.flat_env.jdict["left_hip_z"].set_state(jointsRef["leftHipZ"], jointsVelRef["leftHipZ"])

    def incFrame(self, inc):
        self.frame = (self.frame + inc) % (self.max_frame - 1)

    def reset(self):
        # Insialisasi dengan posisi awal random sesuai referensi
        return self.resetFromFrame(
            self.rng.integers(0, self.max_frame - 5), self.rng.integers(-5, 5)
        )

    def resetFromFrame(self, frame, resetYaw=0):
        self.flat_env.reset()

        self.cur_timestep = 0

        # randomX = self.rng.integers(-20, 20)
        # randomY = self.rng.integers(-20, 20)
        randomX = 1e3
        randomY = 0
        self.targetHighLevel = np.array([randomX, randomY, 0])
        self.flat_env.walk_target_x = randomX
        self.flat_env.walk_target_y = randomY

        # randomProgress = self.rng.random() * 0.5
        randomProgress = 0
        self.flat_env.parts["torso"].reset_position(
            [randomX * randomProgress, randomY * randomProgress, 1.15]
        )
        # self.flat_env.parts["torso"].reset_orientation(
        #     R.from_euler("z", np.rad2deg(np.arctan2(randomY, randomX)) + resetYaw, degrees=True).as_quat()
        # )

        self.frame = frame
        self.frame_update_cnt = 0
        self.setJointsOrientation(self.frame)

        self.initReward()

        self.cur_obs = self.flat_env.robot.calc_state()

        self.incFrame(self.skipFrame)

        return self.getLowLevelObs()

    def resetNonFrame(self):
        self.cur_obs = self.flat_env.reset()

        self.cur_timestep = 0

        self.frame = 0
        self.frame_update_cnt = 0

        self.initReward()

        return self.cur_obs

    def getLowLevelObs(self):
        # Berikan low level agent nilai joint dan target high level
        # deltaZ = self.cur_obs[0]
        # robotInfo = self.cur_obs[3 : 3 + 5]
        # jointVal = self.cur_obs[8 : 8 + 17 * 2]

        # deltaRobotTarget = self.targetHighLevel - self.flat_env.parts[
        #     "torso"
        # ].get_position()

        # endPointRef = self.end_point_df.iloc[self.frame]
        # base_pos = self.flat_env.parts["lwaist"].get_position()
        # r = R.from_euler(
        #     "z", np.arctan2(self.targetHighLevel[1], self.targetHighLevel[0])
        # )

        # endPointObs = []
        # for epMap in self.end_point_map:
        #     # v1 = self.flat_env.parts[epMap].get_position()
        #     v2 = base_pos + r.apply(getJointPos(endPointRef, self.end_point_map[epMap]))
        #     # endPointObs.append(v1)
        #     endPointObs.append(v2)
        # endPointObs = np.array(endPointObs).ravel()

        jointTargetObs = []
        jointsRelRef = self.joints_rel_df.iloc[self.frame]
        jointsVelRef = self.joints_vel_df.iloc[self.frame]
        for jMap in self.joint_map:
            jointTargetObs.append(jointsRelRef[self.joint_map[jMap]])
            jointTargetObs.append(jointsVelRef[self.joint_map[jMap]])
        jointTargetObs = np.array(jointTargetObs)

        # return np.hstack(
        #     (deltaZ, robotInfo, jointVal, deltaRobotTarget[0], deltaRobotTarget[1], jointTargetObs)
        # )
        return np.hstack((self.cur_obs, jointTargetObs))

    def step(self, action):
        return self.low_level_step(action)

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

        score = np.exp(-1 * deltaJoints / self.joint_weight_sum)
        return (score * 3) - 1.8
        # return deltaJoints / self.joint_weight_sum

    def calcJointVelScore(self):
        deltaVel = 0
        jointsVelRef = self.joints_vel_df.iloc[self.frame]

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
        endPointRef = self.end_point_df.iloc[self.frame]

        base_pos = self.flat_env.parts["lwaist"].get_position()
        r = R.from_euler(
            "z", np.arctan2(self.targetHighLevel[1], self.targetHighLevel[0])
        )

        for epMap in self.end_point_map:
            v1 = self.flat_env.parts[epMap].get_position()
            v2 = base_pos + r.apply(getJointPos(endPointRef, self.end_point_map[epMap]))
            # drawLine(base_pos, v2, [1, 0, 0])
            deltaVec = v2 - v1
            deltaEndPoint += np.linalg.norm(deltaVec) * self.end_point_weight[epMap]

        score = np.exp(-2 * deltaEndPoint / self.end_point_weight_sum)
        return score * 3 - 1.6
        # return 15 * np.exp(-30 * deltaEndPoint / self.end_point_weight_sum)
        # return deltaEndPoint / self.end_point_weight_sum

    def calcLowLevelTargetScore(self):
        robotPos = self.flat_env.parts["torso"].get_position()
        robotPos[2] = 0
        deltaRobotTarget = self.targetHighLevel - robotPos

        # Hitung jarak
        distRobotTargetHL = np.linalg.norm(deltaRobotTarget)

        if distRobotTargetHL <= 1.0:
            randomX = self.rng.integers(-20, 20)
            randomY = self.rng.integers(-20, 20)
            self.targetHighLevel = robotPos + np.array([randomX, randomY, 0])
            self.flat_env.walk_target_x = self.targetHighLevel[0]
            self.flat_env.walk_target_y = self.targetHighLevel[1]

        return np.exp(-1 * distRobotTargetHL)

    def calcJumpReward(self, obs):
        return 0

    def calcAliveReward(self):
        # Didapat dari perhitungan reward alive env humanoid
        z = self.cur_obs[0] + self.flat_env.robot.initial_z
        return +2 if z > 0.78 else -1
        
    def low_level_step(self, action):
        # Step di env yang sebenarnya
        f_obs, f_rew, f_done, _ = self.flat_env.step(action)

        self.cur_obs = f_obs

        # Hitung masing masing reward
        self.deltaJoints = self.calcJointScore()  # Rentang -1 - 1
        self.deltaVelJoints = self.calcJointVelScore()
        self.deltaEndPoints = self.calcEndPointScore()  # Rentang -1 - 1
        # self.deltaEndPoints = 0
        self.baseReward = f_rew
        self.lowTargetScore = self.calcLowLevelTargetScore()
        self.aliveReward = self.calcAliveReward()

        jumpReward = self.calcJumpReward(f_obs)  # Untuk task lompat

        reward = [
            self.baseReward,
            self.deltaJoints,
            self.deltaVelJoints,
            self.deltaEndPoints,
            self.lowTargetScore,
            self.aliveReward,
            jumpReward,
        ]
        rewardWeight = [0, 0.25, 0.25, 0.25, 0.1, 1, 0.0]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        # if self.deltaJoints >= 0.4 and self.deltaEndPoints >= 0.2:
        # if self.deltaJoints >= 0:
        #     self.incFrame(self.skipFrame)
        #     self.frame_update_cnt = 0
        # else:
        #     self.frame_update_cnt += 1
        #     if self.frame_update_cnt > 20:
        #         self.incFrame(self.skipFrame)
        #         self.frame_update_cnt = 0
        self.incFrame(self.skipFrame)

        obs = self.getLowLevelObs()

        done = f_done
        self.cur_timestep += 1
        if self.cur_timestep >= self.max_timestep:
            done = True
            self.cur_timestep = 0

        return obs, totalReward, done, {}
