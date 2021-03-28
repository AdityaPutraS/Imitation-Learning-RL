import gym
import pybullet
import pybullet_envs
import pybullet_data
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


class LowLevelHumanoidEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        self.flat_env = gym.make("HumanoidBulletEnv-v0")

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=[1 + 5 + 17 * 2 + 2]
        )
        self.action_space = self.flat_env.action_space

        self.joints_df = pd.read_csv("Processed Joints CSV/walk08_03JointPosRad.csv")
        self.end_point_df = pd.read_csv(
            "Processed Joints CSV/walk08_03JointVecFromHip.csv"
        )

        self.frame = 0
        self.frame_update_cnt = 0
        self.max_frame = (
            125  # Untuk 08_03, frame 0 - 125 merupakan siklus (2 - 127 di blender)
        )

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
        self.frame = (self.frame + inc) % (self.max_frame - 1)

    def reset(self):
        # Insialisasi dengan posisi awal random sesuai referensi
        return self.resetFromFrame(
            self.rng.integers(0, self.max_frame - 5), self.rng.integers(-15, 15)
        )

    def resetFromFrame(self, frame, resetYaw=0):
        self.flat_env.reset()

        randomX = self.rng.integers(-20, 20)
        randomY = self.rng.integers(-20, 20)
        self.targetHighLevel = np.array([randomX, randomY, 0])
        self.flat_env.walk_target_x = randomX
        self.flat_env.walk_target_y = randomY

        randomProgress = self.rng.random() * 0.5
        self.flat_env.parts["torso"].reset_position(
            [randomX * randomProgress, randomY * randomProgress, 1.15]
        )
        self.flat_env.parts["torso"].reset_orientation(
            R.from_euler("z", resetYaw, degrees=True).as_quat()
        )

        self.frame = frame
        self.frame_update_cnt = 0
        self.setJointsOrientation(self.frame)

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0
        self.lowTargetScore = 0
        self.highTargetScore = 0

        self.cur_obs = self.flat_env.robot.calc_state()

        self.incFrame(self.skipFrame)

        return self.getLowLevelObs()

    def resetNonFrame(self):
        self.cur_obs = self.flat_env.reset()

        self.frame = 0
        self.frame_update_cnt = 0

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0
        self.lowTargetScore = 0
        self.highTargetScore = 0

        return self.cur_obs

    def getLowLevelObs(self):
        # Berikan low level agent nilai joint dan target high level
        deltaZ = self.cur_obs[0]
        robotInfo = self.cur_obs[3 : 3 + 5]
        jointVal = self.cur_obs[8 : 8 + 17 * 2]

        deltaRobotTarget = self.targetHighLevel = self.flat_env.parts[
            "torso"
        ].get_position()

        return np.hstack(
            (deltaZ, robotInfo, jointVal, deltaRobotTarget[0], deltaRobotTarget[1])
        )

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

        return np.exp(-5 * deltaJoints / self.joint_weight_sum)

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
            deltaVec = v2 - v1
            deltaEndPoint += np.linalg.norm(deltaVec) * self.end_point_weight[epMap]

        return np.exp(-10 * deltaEndPoint / self.end_point_weight_sum)

    def calcLowLevelTargetScore(self):
        robotPos = self.flat_env.parts["torso"].get_position()
        robotPos[2] = 0
        deltaRobotTarget = self.targetHighLevel - robotPos

        # Hitung jarak
        distRobotTargetHL = np.linalg.norm(deltaRobotTarget)

        if distRobotTargetHL <= 1.0:
            randomX = self.rng.integers(0, 20)
            randomY = self.rng.integers(-20, 20)
            self.targetHighLevel = robotPos + np.array([randomX, randomY, 0])
            self.flat_env.walk_target_x = self.targetHighLevel[0]
            self.flat_env.walk_target_y = self.targetHighLevel[1]

        return np.exp(-1 * distRobotTargetHL)

    def calcJumpReward(self, obs):
        return 0

    def low_level_step(self, action):
        # Step di env yang sebenarnya
        f_obs, f_rew, f_done, _ = self.flat_env.step(action)

        self.cur_obs = f_obs

        # Hitung masing masing reward
        self.deltaJoints = self.calcJointScore()  # Rentang 0 - 1
        self.deltaEndPoints = self.calcEndPointScore()  # Rentang 0 - 1
        self.baseReward = f_rew
        self.lowTargetScore = self.calcLowLevelTargetScore()

        jumpReward = self.calcJumpReward(f_obs)  # Untuk task lompat

        reward = [
            self.baseReward,
            self.deltaJoints,
            self.deltaEndPoints,
            self.lowTargetScore,
            jumpReward,
        ]
        rewardWeight = [0.125, 0.25, 0.25, 0.375, 0.0]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        obs = self.getLowLevelObs()

        if self.deltaJoints >= 0.15 and self.deltaEndPoints >= 0.09:
            self.incFrame(self.skipFrame)
            self.frame_update_cnt = 0
        else:
            self.frame_update_cnt += 1
            if self.frame_update_cnt > 10:
                self.incFrame(self.skipFrame)
                self.frame_update_cnt = 0

        return obs, totalReward, f_done, {}
