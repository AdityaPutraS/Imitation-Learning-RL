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


def drawLine(c1, c2, color, lifeTime=0.1):
    return pybullet.addUserDebugLine(
        c1, c2, lineColorRGB=color, lineWidth=5, lifeTime=lifeTime
    )


class LowLevelHumanoidEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        self.flat_env = HumanoidBulletEnv()

        # self.observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=[1 + 5 + 17 * 2 + 2 + 8]
        # )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=[42 + 8 * 2]
        )  # Foot contact tidak dimasukan
        self.action_space = self.flat_env.action_space

        reference_name = "motion09_03"
        basePath = "~/GitHub/TA/Relative_Joints_CSV"
        self.joints_df = pd.read_csv(
            "{}/{}JointPosRad.csv".format(basePath, reference_name)
        )
        self.joints_vel_df = pd.read_csv(
            "{}/{}JointSpeedRadSec.csv".format(basePath, reference_name)
        )
        self.joints_rel_df = pd.read_csv(
            "{}/{}JointPosRadRelative.csv".format(basePath, reference_name)
        )
        self.end_point_df = pd.read_csv(
            "{}/{}JointVecFromHip.csv".format(basePath, reference_name)
        )

        self.cur_timestep = 0
        self.max_timestep = 3000

        self.frame = 0

        # Untuk 08_03, frame 0 - 125 merupakan siklus (2 - 127 di blender)
        # Untuk 09_01, frame 0 - 90 merupakan siklus (1 - 91 di blender)
        # Untuk 02_04, frame 0 - 298 (2 - 300 di blender)
        self.max_frame = (
            len(self.joints_df) - 1
        )  # Minus 1 karena velocity merupakan delta position

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

        self.target = np.array([1, 0, 0])
        self.targetLen = 5
        self.targetHighLevel = np.array([0, 0, 0])
        self.targetHighLevelLen = 1000
        self.skipFrame = 1

        self.starting_ep_pos = np.array([0, 0, 0])
        self.robot_pos = np.array([0, 0, 0])

        self.frame_update_cnt = 0

        self.last_robotPos = np.array([0, 0, 0])

    def initReward(self):
        self.deltaJoints = 0
        self.deltaVelJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0
        self.lowTargetScore = 0
        self.last_lowTargetScore = 0
        self.highTargetScore = 0
        self.aliveReward = 0
        self.electricityScore = 0
        self.jointLimitScore = 0

        self.delta_deltaJoints = 0
        self.delta_deltaVelJoints = 0
        self.delta_deltaEndPoints = 0
        self.delta_lowTargetScore = 0
        self.delta_highTargetScore = 0

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
        # self.frame_update_cnt = (self.frame_update_cnt + 1) % 2
        # if(self.frame_update_cnt == 0):
        self.frame = (self.frame + inc) % (self.max_frame - 1)

        if self.frame == 0:
            self.starting_ep_pos = self.robot_pos.copy()

    def reset(self):
        # Insialisasi dengan posisi awal random sesuai referensi
        # return self.resetWithYaw(self.rng.integers(-10, 10))
        # return self.resetFromFrame(startFrame=self.rng.integers(0, self.max_frame-5), resetYaw=self.rng.integers(-180, 180))
        return self.resetFromFrame(
            startFrame=self.rng.integers(0, self.max_frame - 5), resetYaw=0
        )

    def setWalkTarget(self, x, y):
        self.flat_env.walk_target_x = x
        self.flat_env.walk_target_y = y
        self.flat_env.robot.walk_target_x = x
        self.flat_env.robot.walk_target_y = y

    def reassignWalkTarget(self):
        vTargetRobot = self.target - self.robot_pos
        degTargetRobot = np.arctan2(vTargetRobot[1], vTargetRobot[0])
        lenTargetRobot = np.linalg.norm(vTargetRobot)
        _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()
        degLimit = np.deg2rad(135)
        if degTargetRobot - yaw > degLimit:
            vTargetRobot = (
                np.array([np.cos(degLimit + yaw), np.sin(degLimit + yaw), 0])
                * lenTargetRobot
            )
        elif degTargetRobot - yaw < -degLimit:
            vTargetRobot = (
                np.array([np.cos(-degLimit + yaw), np.sin(-degLimit + yaw), 0])
                * lenTargetRobot
            )

        self.targetHighLevel = vTargetRobot

        vTargetRobot = vTargetRobot / lenTargetRobot
        walkTarget = self.robot_pos + vTargetRobot * (
            self.targetHighLevelLen - (self.targetLen - lenTargetRobot)
        )
        self.setWalkTarget(walkTarget[0], walkTarget[1])

    def getRandomVec(self, vecLen, z, initYaw=0):
        # randomRad = initYaw + np.deg2rad(self.rng.integers(-180, 180))
        randomRad = initYaw + np.deg2rad(self.rng.integers(-180, 180))
        randomX = np.cos(randomRad) * vecLen
        randomY = np.sin(randomRad) * vecLen

        return np.array([randomX, randomY, z])

    def resetFromFrame(self, startFrame=0, resetYaw=0):
        self.flat_env.reset()

        self.cur_timestep = 0

        self.target = self.getRandomVec(self.targetLen, 0)
        # self.target = np.array([4, 0, 0])

        self.frame = startFrame
        self.setJointsOrientation(self.frame)

        # Posisi awal robot
        # robotPos = self.getRandomVec(3, 1.15)
        robotPos = np.array([0, 0, 1.15])
        self.robot_pos = np.array([robotPos[0], robotPos[1], 0])
        self.last_robotPos = self.robot_pos.copy()
        self.reassignWalkTarget()
        self.flat_env.robot.robot_body.reset_position(robotPos)

        degToTarget = np.rad2deg(
            np.arctan2(self.targetHighLevel[1], self.targetHighLevel[0])
        )
        robotRot = R.from_euler("z", degToTarget + resetYaw, degrees=True)
        self.flat_env.robot.robot_body.reset_orientation(robotRot.as_quat())

        # Gunakan kaki kanan sebagai acuan
        rightFootPosActual = self.flat_env.parts["right_foot"].get_position()
        rightFootPosActual[2] = 0
        rotDeg = R.from_euler("z", degToTarget, degrees=True)
        rightFootPosRef = rotDeg.apply(
            getJointPos(
                self.end_point_df.iloc[self.frame], self.end_point_map["right_foot"]
            )
        )
        rightFootPosRef[2] = 0
        # Pilih hips pos agar starting_ep_pos + rightFootPosRef == rightFootPosActual
        # starting_ep_pos = rightFootPosActual - rightFootPosRef
        self.starting_ep_pos = rightFootPosActual - rightFootPosRef

        rightLegPosRef = rotDeg.apply(
            getJointPos(self.end_point_df.iloc[self.frame], "RightLeg")
        )
        rightLegPosRefNext = rotDeg.apply(
            getJointPos(self.end_point_df.iloc[self.frame + 1], "RightLeg")
        )
        startingVelocity = (rightLegPosRefNext - rightLegPosRef) / 0.0165
        self.flat_env.robot.robot_body.reset_velocity(linearVelocity=startingVelocity)

        self.initReward()

        drawLine(self.robot_pos, self.target, [1, 0, 0], lifeTime=0)

        self.frame_update_cnt = 0
        self.incFrame(self.skipFrame)

        self.cur_obs = self.flat_env.robot.calc_state()
        return self.getLowLevelObs()

    def getLowLevelObs(self):
        jointTargetObs = []
        jointsRelRef = self.joints_rel_df.iloc[self.frame]
        jointsVelRef = self.joints_vel_df.iloc[self.frame]
        for jMap in self.joint_map:
            jointTargetObs.append(jointsRelRef[self.joint_map[jMap]])
            jointTargetObs.append(jointsVelRef[self.joint_map[jMap]])
        jointTargetObs = np.array(jointTargetObs)

        targetInfo = self.cur_obs[1:3]
        jointInfo = self.cur_obs[8 : 8 + 17 * 2]

        # return np.hstack((targetInfo, jointInfo, jointTargetObs))
        return np.hstack((self.cur_obs[:-2], jointTargetObs))

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

        # score = np.exp(-1 * deltaJoints / self.joint_weight_sum)
        # return (score * 3) - 2.3
        # return deltaJoints / self.joint_weight_sum
        return -deltaJoints / self.joint_weight_sum

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

        # score = np.exp(-1 * deltaVel / self.joint_vel_weight_sum)
        # return (score * 3) - 1.8
        return -deltaVel / self.joint_vel_weight_sum

    def calcEndPointScore(self, debug=False):
        deltaEndPoint = 0
        if debug:
            deltaEndPoint = []
        endPointRef = self.end_point_df.iloc[self.frame]

        # base_pos = self.flat_env.parts["lwaist"].get_position()
        # base_pos[2] = 1
        r = R.from_euler(
            "z", np.arctan2(self.targetHighLevel[1], self.targetHighLevel[0])
        )

        for epMap in self.end_point_map:
            v1 = self.flat_env.parts[epMap].get_position()
            # v2 = base_pos + r.apply(getJointPos(endPointRef, self.end_point_map[epMap]))
            v2 = self.starting_ep_pos + r.apply(
                getJointPos(endPointRef, self.end_point_map[epMap])
            )
            # drawLine(v1, v2, [1, 0, 0])
            deltaVec = v2 - v1
            dist = np.linalg.norm(deltaVec)
            if debug:
                deltaEndPoint.append(dist * self.end_point_weight[epMap])
            else:
                deltaEndPoint += dist * self.end_point_weight[epMap]

        if debug:
            return deltaEndPoint
        else:
            # score = np.exp(-10 * (deltaEndPoint/self.end_point_weight_sum))
            # return 2 * score - 0.5
            return -deltaEndPoint / self.end_point_weight_sum

    def calcJumpReward(self, obs):
        return 0

    def calcAliveReward(self):
        # Didapat dari perhitungan reward alive env humanoid
        z = self.cur_obs[0] + self.flat_env.robot.initial_z
        # return +2 if z > 0.78 else -1
        return +2 if z > 0.75 else -1

    def calcElectricityCost(self, action):
        runningCost = -1.0 * float(
            np.abs(action * self.flat_env.robot.joint_speeds).mean()
        )
        stallCost = -0.1 * float(np.square(action).mean())
        return runningCost + stallCost

    def calcJointLimitCost(self):
        return -0.1 * self.flat_env.robot.joints_at_limit

    def calcLowLevelTargetScore(self):
        # Hitung jarak
        distRobotTargetHL = np.linalg.norm(self.target - self.robot_pos)
        # return 2 * np.exp(-1 * distRobotTargetHL / 2) + self.last_lowTargetScore
        return -distRobotTargetHL

    def checkTarget(self):
        distToTarget = np.linalg.norm(self.robot_pos - self.target)

        if distToTarget <= 1:
            _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()
            randomTarget = self.getRandomVec(self.targetLen, 0, initYaw=yaw)
            newTarget = self.robot_pos + randomTarget
            drawLine(self.target, newTarget, [1, 0, 0], lifeTime=0)
            self.target = newTarget
            self.starting_ep_pos = self.robot_pos.copy()
            self.last_lowTargetScore = self.lowTargetScore

        self.reassignWalkTarget()

    def drawDebugRobotPosLine(self):
        if self.cur_timestep % 10 == 0:
            drawLine(self.last_robotPos, self.robot_pos, [1, 1, 1], lifeTime=0)
            self.last_robotPos = self.robot_pos.copy()

    def updateReward(self, action):
        jointScore = self.calcJointScore()
        jointVelScore = self.calcJointVelScore()
        endPointScore = self.calcEndPointScore()
        lowTargetScore = self.calcLowLevelTargetScore()

        self.delta_deltaJoints = (jointScore - self.deltaJoints) / 0.0165
        self.delta_deltaVelJoints = (jointVelScore - self.deltaVelJoints) / 0.0165 * 0.1
        self.delta_deltaEndPoints = (endPointScore - self.deltaEndPoints) / 0.0165
        self.delta_lowTargetScore = (
            (lowTargetScore - self.lowTargetScore) / 0.0165 * 0.1
        )

        self.deltaJoints = jointScore
        self.deltaVelJoints = jointVelScore
        self.deltaEndPoints = endPointScore
        self.lowTargetScore = lowTargetScore
        # self.baseReward = base_reward
        self.electricityScore = self.calcElectricityCost(action)
        self.jointLimitScore = self.calcJointLimitCost()
        self.aliveReward = self.calcAliveReward()

    def low_level_step(self, action):
        # Step di env yang sebenarnya
        # f_obs, f_rew, f_done, _ = self.flat_env.step(action)
        self.flat_env.robot.apply_action(action)
        self.flat_env.scene.global_step()

        f_obs = self.flat_env.robot.calc_state()  # also calculates self.joints_at_limit

        body_xyz = self.flat_env.robot.body_xyz
        self.robot_pos[0] = body_xyz[0]
        self.robot_pos[1] = body_xyz[1]
        self.robot_pos[2] = 0

        self.cur_obs = f_obs

        self.updateReward(action=action)

        reward = [
            self.baseReward,
            self.delta_deltaJoints,
            self.delta_deltaVelJoints,
            # self.delta_deltaEndPoints,
            self.deltaEndPoints,
            self.delta_lowTargetScore,
            self.electricityScore,
            self.jointLimitScore,
            self.aliveReward,
        ]

        rewardWeight = [0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        self.incFrame(self.skipFrame)

        self.checkTarget()

        self.drawDebugRobotPosLine()

        obs = self.getLowLevelObs()

        done = self.aliveReward < 0
        self.cur_timestep += 1
        if self.cur_timestep >= self.max_timestep:
            done = True

        return obs, totalReward, done, {}
