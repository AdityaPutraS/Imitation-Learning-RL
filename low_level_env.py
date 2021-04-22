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
        self.highLevelDegTarget = 0

        # self.predefinedTarget = np.array([
        #     [5, 0, 0],
        #     [5, -5, 0],
        #     [0, -5, 0],
        #     [0, 0, 0]
        # ])
        self.predefinedTarget = np.array([
            [5, -1, 0],
            [0, -2, 0],
            [5, -3, 0],
            [0, -4, 0],
            [0, 0, 0]
        ])
        self.predefinedTargetIndex = 0
        self.usePredefinedTarget=False

        self.skipFrame = 1

        self.starting_ep_pos = np.array([0, 0, 0])
        self.robot_pos = np.array([0, 0, 0])

        self.frame_update_cnt = 0

        self.last_robotPos = np.array([0, 0, 0])

        self.initReward()

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
        self.bodyPostureScore = 0

        self.delta_deltaJoints = 0
        self.delta_deltaVelJoints = 0
        self.delta_deltaEndPoints = 0
        self.delta_lowTargetScore = 0
        self.delta_highTargetScore = 0
        self.delta_bodyPostureScore = 0

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

    def getRandomVec(self, vecLen, z, initYaw=0):
        # randomRad = initYaw + np.deg2rad(self.rng.integers(-180, 180))
        randomRad = initYaw + np.deg2rad(self.rng.integers(-180, 180))
        randomX = np.cos(randomRad) * vecLen
        randomY = np.sin(randomRad) * vecLen

        return np.array([randomX, randomY, z])

    def resetFromFrame(self, startFrame=0, resetYaw=0):
        self.flat_env.reset()

        self.cur_timestep = 0

        # Init target
        if(self.usePredefinedTarget):
            self.predefinedTargetIndex = 0
            self.target = self.predefinedTarget[self.predefinedTargetIndex].copy()
        else:
            self.target = self.getRandomVec(self.targetLen, 0)

        self.frame = startFrame
        self.setJointsOrientation(self.frame)

        # Posisi awal robot
        # robotPos = self.getRandomVec(3, 1.15)
        robotPos = np.array([0, 0, 1.15])
        self.robot_pos = np.array([robotPos[0], robotPos[1], 0])
        self.last_robotPos = self.robot_pos.copy()
        self.flat_env.robot.robot_body.reset_position(robotPos)

        degToTarget = np.rad2deg(np.arctan2(self.target[1], self.target[0]))
        self.setWalkTarget(np.cos(degToTarget) * 1000, np.sin(degToTarget) * 1000)
        robotRot = R.from_euler("z", degToTarget + resetYaw, degrees=True)
        self.flat_env.robot.robot_body.reset_orientation(robotRot.as_quat())

        self.highLevelDegTarget = np.deg2rad(degToTarget)

        endPointRef = self.end_point_df.iloc[self.frame]
        endPointRefNext = self.end_point_df.iloc[self.frame + 1]

        # Gunakan kaki kanan sebagai acuan
        rightFootPosActual = self.flat_env.parts["right_foot"].get_position()
        rightFootPosActual[2] = 0
        rotDeg = R.from_euler("z", degToTarget, degrees=True)
        rightFootPosRef = rotDeg.apply(
            getJointPos(endPointRef, self.end_point_map["right_foot"])
        )
        rightFootPosRef[2] = 0
        # Pilih hips pos agar starting_ep_pos + rightFootPosRef == rightFootPosActual
        # starting_ep_pos = rightFootPosActual - rightFootPosRef
        self.starting_ep_pos = rightFootPosActual - rightFootPosRef

        rightLegPosRef = rotDeg.apply(getJointPos(endPointRef, "RightLeg"))
        rightLegPosRefNext = rotDeg.apply(getJointPos(endPointRefNext, "RightLeg"))
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

    def calcJointScore(self, useExp=False):
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

        score = -deltaJoints / self.joint_weight_sum
        if(useExp):
            score = np.exp(score)
            return (score * 3) - 2.3
        return score

    def calcJointVelScore(self, useExp=False):
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

        score = -deltaVel / self.joint_vel_weight_sum
        if(useExp):
            score = np.exp(score)
            return (score * 3) - 1.8
        return score

    def calcEndPointScore(self, useExp=False):
        deltaEndPoint = 0
        endPointRef = self.end_point_df.iloc[self.frame]

        # base_pos = self.flat_env.parts["lwaist"].get_position()
        # base_pos[2] = 1
        r = R.from_euler("z", self.highLevelDegTarget)

        for epMap in self.end_point_map:
            v1 = self.flat_env.parts[epMap].get_position()
            # v2 = base_pos + r.apply(getJointPos(endPointRef, self.end_point_map[epMap]))
            v2 = self.starting_ep_pos + r.apply(
                getJointPos(endPointRef, self.end_point_map[epMap])
            )
            # drawLine(v1, v2, [1, 0, 0])
            deltaVec = v2 - v1
            dist = np.linalg.norm(deltaVec)
            deltaEndPoint += dist * self.end_point_weight[epMap]

        score = -deltaEndPoint / self.end_point_weight_sum
        if(useExp):
            score = np.exp(10 * score)
            return 2 * score - 0.5
        return score

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
    
    def calcBodyPostureScore(self, useExp=False):
        roll, pitch, yaw = self.flat_env.robot.robot_body.pose().rpy()
        score = -np.abs(yaw - self.highLevelDegTarget) + (-roll) + (-pitch)
        if(useExp):
            score = np.exp(score)
        return score

    def checkTarget(self):
        distToTarget = np.linalg.norm(self.robot_pos - self.target)

        if distToTarget <= 0.5:
            _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()
            randomTarget = self.getRandomVec(self.targetLen, 0, initYaw=yaw)
            newTarget = self.robot_pos + randomTarget
            if(self.usePredefinedTarget):
                self.predefinedTargetIndex = (self.predefinedTargetIndex + 1) % len(self.predefinedTarget)
                newTarget = self.predefinedTarget[self.predefinedTargetIndex]
            drawLine(self.target, newTarget, [1, 0, 0], lifeTime=0)
            self.target = newTarget
            self.starting_ep_pos = self.robot_pos.copy()

        # Reassign highLevelDegTarget
        vRobotTarget = self.target - self.robot_pos
        self.highLevelDegTarget = np.arctan2(vRobotTarget[1], vRobotTarget[0])
        self.setWalkTarget(self.robot_pos[0] + np.cos(self.highLevelDegTarget) * 10, self.robot_pos[1] + np.sin(self.highLevelDegTarget) * 10)

    def drawDebugRobotPosLine(self):
        if self.cur_timestep % 10 == 0:
            drawLine(self.last_robotPos, self.robot_pos, [1, 1, 1], lifeTime=0)
            self.last_robotPos = self.robot_pos.copy()

    def updateReward(self, action):
        jointScore = self.calcJointScore(useExp=True)
        jointVelScore = self.calcJointVelScore(useExp=True)
        endPointScore = self.calcEndPointScore(useExp=True)
        lowTargetScore = self.calcLowLevelTargetScore()
        bodyPostureScore = self.calcBodyPostureScore(useExp=True)

        self.delta_deltaJoints = (jointScore - self.deltaJoints) / 0.0165
        self.delta_deltaVelJoints = (jointVelScore - self.deltaVelJoints) / 0.0165 * 0.1
        self.delta_deltaEndPoints = (endPointScore - self.deltaEndPoints) / 0.0165
        self.delta_lowTargetScore = (
            (lowTargetScore - self.lowTargetScore) / 0.0165 * 0.1
        )
        self.delta_bodyPostureScore = (bodyPostureScore - self.bodyPostureScore) / 0.0165 * 0.1

        self.deltaJoints = jointScore
        self.deltaVelJoints = jointVelScore
        self.deltaEndPoints = endPointScore
        self.lowTargetScore = lowTargetScore
        # self.baseReward = base_reward
        self.electricityScore = self.calcElectricityCost(action)
        self.jointLimitScore = self.calcJointLimitCost()
        self.aliveReward = self.calcAliveReward()
        self.bodyPostureScore = bodyPostureScore

    def checkIfDone(self):
        isAlive = self.aliveReward > 0
        isNearTarget = np.linalg.norm(self.target - self.robot_pos) <= self.targetLen + 1
        return not(isAlive and isNearTarget)

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
            self.delta_deltaJoints,
            self.delta_deltaVelJoints,
            self.delta_deltaEndPoints,
            self.delta_lowTargetScore,
            self.electricityScore,
            self.jointLimitScore,
            self.aliveReward,
            self.delta_bodyPostureScore,
        ]

        rewardWeight = [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        self.incFrame(self.skipFrame)

        self.checkTarget()

        self.drawDebugRobotPosLine()

        obs = self.getLowLevelObs()

        done = self.checkIfDone()
        self.cur_timestep += 1
        if self.cur_timestep >= self.max_timestep:
            done = True

        return obs, totalReward, done, {}
