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
from math_util import rotFrom2Vec, projPointLineSegment

logger = logging.getLogger(__name__)


def getJointPos(df, joint, multiplier=1):
    x = df[joint + "_Xposition"]
    y = df[joint + "_Yposition"]
    z = df[joint + "_Zposition"]
    return np.array([x, y, z]) * multiplier


def drawLine(c1, c2, color, lifeTime=0.1, width=5):
    return pybullet.addUserDebugLine(
        c1, c2, lineColorRGB=color, lineWidth=width, lifeTime=lifeTime
    )


class HierarchicalHumanoidEnv(MultiAgentEnv):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        self.flat_env = HumanoidBulletEnv()
        # self.action_space = self.flat_env.action_space
        # self.observation_space = self.flat_env.observation_space

        self.motion_list = ["motion08_03", "motion09_03"]

        # self.high_level_obs_space = Box(low=-np.inf, high=np.inf, shape=[2 + 42])
        self.high_level_obs_space = Box(
            low=-np.inf, high=np.inf, shape=[2 + 42 + 8 * 2]
        )
        # self.high_level_act_space = Box(low=-1, high=1, shape=[2])  # cos(target), sin(target)
        self.high_level_act_space = Box(
            low=-1, high=1, shape=[2 + 17 * 2]
        )  # Sudut & Velocity setiap sendi referensi

        # self.low_level_obs_space = Box(low=-np.inf, high=np.inf, shape=[42 + 8 * 2])
        self.low_level_obs_space = Box(low=-np.inf, high=np.inf, shape=[4 + 17 * 2 + 17 * 2])
        self.low_level_act_space = self.flat_env.action_space

        self.step_per_level = 20
        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0
        basePath = "~/GitHub/TA/Relative_Joints_CSV"
        self.joints_df = [
            pd.read_csv("{}/{}JointPosRad.csv".format(basePath, mot))
            for mot in self.motion_list
        ]

        self.joints_vel_df = [
            pd.read_csv("{}/{}JointSpeedRadSec.csv".format(basePath, mot))
            for mot in self.motion_list
        ]

        self.joints_rel_df = [
            pd.read_csv("{}/{}JointPosRadRelative.csv".format(basePath, mot))
            for mot in self.motion_list
        ]

        self.end_point_df = [
            pd.read_csv("{}/{}JointVecFromHip.csv".format(basePath, mot))
            for mot in self.motion_list
        ]

        self.cur_timestep = 0
        self.max_timestep = 3000

        self.frame = 0
        self.frame_update_cnt = 0

        self.max_frame = [len(df) - 1 for df in self.joints_df]

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

        self.target = np.array([1, 0, 0])
        self.targetLen = 5
        self.highLevelDegTarget = 0  # Sudut yang harus dicapai oleh robot (radian)

        self.predefinedTarget = np.array([[]])
        self.predefinedTargetIndex = 0
        self.usePredefinedTarget = False

        self.skipFrame = 5

        # Menandakan motion apa yang harus dijalankan sekarang dan pada frame berapa
        self.selected_motion = 1
        self.selected_motion_frame = 0

        self.starting_ep_pos = np.array([0, 0, 0])
        self.starting_robot_pos = np.array([0, 0, 0])
        self.robot_pos = np.array([0, 0, 0])

        self.frame_update_cnt = 0

        self.last_robotPos = np.array([0, 0, 0])

        # Target sementara (sudut & kecepatan gerak referensi)
        self.jointTarget = [0] * 16

        self.initReward()

    def initReward(self):
        self.deltaJoints = 0
        self.deltaVelJoints = 0
        self.deltaJoints_low = 0
        self.deltaVelJoints_low = 0
        self.deltaEndPoints = 0
        self.baseReward = 0
        self.lowTargetScore = -self.targetLen
        self.aliveReward = 0
        self.electricityScore = 0
        self.jointLimitScore = 0
        self.bodyPostureScore = 0

        self.highTargetScore = -self.targetLen
        self.driftScore = 0
        self.cumulative_driftScore = 0

        self.delta_deltaJoints = 0
        self.delta_deltaVelJoints = 0
        self.delta_deltaEndPoints = 0
        self.delta_lowTargetScore = 0
        self.delta_bodyPostureScore = 0

        self.delta_highTargetScore = 0
        self.cumulative_deltaJoints_low = 0
        self.cumulative_deltaVelJoints_low = 0

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
        oldFrame = self.selected_motion_frame
        self.selected_motion_frame = (self.selected_motion_frame + inc) % (
            self.max_frame[self.selected_motion] - 1
        )

        # selected_motion_frame hanya akan lebih kecil dari oldFrame ketika sudah berada di ujung
        if self.selected_motion_frame <= oldFrame:
            self.starting_ep_pos = self.robot_pos.copy()

    def reset(self, startFrame=None, startFromRef=True):
        # Insialisasi dengan posisi awal random sesuai referensi
        return self.resetFromFrame(
            startFrame=self.rng.integers(0, self.max_frame[self.selected_motion] - 5)
            if startFrame == None
            else startFrame,
            resetYaw=self.rng.integers(-180, 180),
            startFromRef=True,
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

    def resetFromFrame(self, startFrame=0, resetYaw=0, startFromRef=True):
        self.flat_env.reset()

        self.cur_timestep = 0

        # Init target
        if self.usePredefinedTarget:
            self.predefinedTargetIndex = 0
            self.target = self.predefinedTarget[self.predefinedTargetIndex].copy()
        else:
            self.target = self.getRandomVec(self.targetLen, 0)

        self.setWalkTarget(self.target[0], self.target[1])

        if startFromRef:
            self.selected_motion_frame = startFrame
            self.setJointsOrientation(self.selected_motion, self.selected_motion_frame)

        # Posisi awal robot
        # robotPos = self.getRandomVec(3, 1.15)
        robotPos = np.array([0, 0, 1.15])
        self.robot_pos = np.array([robotPos[0], robotPos[1], 0])
        self.last_robotPos = self.robot_pos.copy()
        self.starting_robot_pos = self.robot_pos.copy()
        self.flat_env.robot.robot_body.reset_position(robotPos)

        degToTarget = np.rad2deg(np.arctan2(self.target[1], self.target[0])) + resetYaw
        self.setWalkTarget(np.cos(degToTarget) * self.targetLen, np.sin(degToTarget) * self.targetLen)
        robotRot = R.from_euler("z", degToTarget, degrees=True)
        self.flat_env.robot.robot_body.reset_orientation(robotRot.as_quat())

        self.highLevelDegTarget = np.deg2rad(degToTarget)

        endPointRef = self.end_point_df[self.selected_motion].iloc[
            self.selected_motion_frame
        ]
        endPointRefNext = self.end_point_df[self.selected_motion].iloc[
            self.selected_motion_frame + 1
        ]

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

        if startFromRef:
            rightLegPosRef = rotDeg.apply(getJointPos(endPointRef, "RightLeg"))
            rightLegPosRefNext = rotDeg.apply(getJointPos(endPointRefNext, "RightLeg"))
            startingVelocity = (rightLegPosRefNext - rightLegPosRef) / 0.0165
            self.flat_env.robot.robot_body.reset_velocity(
                linearVelocity=startingVelocity
            )

        self.initReward()

        drawLine(self.robot_pos, self.target, [1, 0, 0], lifeTime=0)

        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0

        self.frame_update_cnt = 0
        self.incFrame(self.skipFrame)

        # self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        self.low_level_agent_id = "low_level_agent"

        self.cur_obs = self.flat_env.robot.calc_state()
        return {"high_level_agent": self.getHighLevelObs()}

    def getLowLevelObs(self):
        # jointTargetObs = []
        # jointsRelRef = self.joints_rel_df[self.selected_motion].iloc[
        #     self.selected_motion_frame
        # ]
        # jointsVelRef = self.joints_vel_df[self.selected_motion].iloc[
        #     self.selected_motion_frame
        # ]
        # for jMap in self.joint_map:
        #     jointTargetObs.append(jointsRelRef[self.joint_map[jMap]])
        #     jointTargetObs.append(jointsVelRef[self.joint_map[jMap]])
        # jointTargetObs = np.array(jointTargetObs)

        # # Tidak usah berikan info feet contact
        # return np.hstack((self.cur_obs[:-2], jointTargetObs))

        targetInfo = self.cur_obs[1:3]
        postureInfo = self.cur_obs[6:8]
        jointInfo = self.cur_obs[8 : 8 + 17 * 2]
        return np.hstack((targetInfo, postureInfo, jointInfo, self.jointTarget))

    def getHighLevelObs(self):
        _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()

        targetTheta = np.arctan2(
            self.target[1] - self.robot_pos[1], self.target[0] - self.robot_pos[0]
        )
        angleToTarget = targetTheta - yaw
        degTarget = [np.cos(angleToTarget), np.sin(angleToTarget)]

        startPosTheta = np.arctan2(
            self.starting_robot_pos[1] - self.robot_pos[1],
            self.starting_robot_pos[0] - self.robot_pos[0],
        )
        angleToStart = startPosTheta - yaw
        degStart = [np.cos(angleToStart), np.sin(angleToStart)]

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

        # Tidak usah berikan info feet contact
        return np.hstack(
            (self.cur_obs[:1], degTarget, degStart, self.cur_obs[3:-2], jointTargetObs)
        )

    def step(self, action_dict, debug=False):
        assert len(action_dict) == 1, action_dict

        body_xyz = self.flat_env.robot.body_xyz
        self.robot_pos[0] = body_xyz[0]
        self.robot_pos[1] = body_xyz[1]
        self.robot_pos[2] = 0

        if "high_level_agent" in action_dict:
            return self.high_level_step(action_dict["high_level_agent"], debug=debug)
        else:
            return self.low_level_step(list(action_dict.values())[0], debug=debug)

    def calcJointScore(self, useExp=False):
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

        score = -deltaJoints / self.joint_weight_sum
        if useExp:
            score = np.exp(4 * score)
        return score

    def calcJointVelScore(self, useExp=False):
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

        score = -deltaVel / self.joint_vel_weight_sum
        if useExp:
            score = np.exp(score / 2)
        return score

    def calcJointPosVelLowScore(self, useExp=True):
        deltaJoints = 0
        deltaVel = 0
        for i in range(17):
            relPos, relVel = self.cur_obs[8 + i * 2], self.cur_obs[8 + i * 2 + 1]
            relVel /= 0.1
            deltaJoints += np.abs(relPos - self.jointTarget[i * 2])
            deltaVel += np.abs(relVel - self.jointTarget[i * 2 + 1])
        scoreJoint = -deltaJoints / 17
        scoreVel = -deltaVel / 17
        if useExp:
            scoreJoint = np.exp(scoreJoint * 2)
            scoreVel = np.exp(scoreVel / 2)
        return scoreJoint, scoreVel

    def calcEndPointScore(self, useExp=False):
        deltaEndPoint = 0
        endPointRef = self.end_point_df[self.selected_motion].iloc[
            self.selected_motion_frame
        ]

        # base_pos = self.flat_env.parts["lwaist"].get_position()
        r = R.from_euler("z", self.highLevelDegTarget)

        for epMap in self.end_point_map:
            v1 = self.flat_env.parts[epMap].get_position()
            # v2 = base_pos + r.apply(getJointPos(endPointRef, self.end_point_map[epMap]))
            v2 = self.starting_ep_pos + r.apply(
                getJointPos(endPointRef, self.end_point_map[epMap])
            )
            # drawLine(v1, v2, [1, 0, 0])
            deltaVec = v2 - v1
            deltaEndPoint += np.linalg.norm(deltaVec) * self.end_point_weight[epMap]

        score = -deltaEndPoint / self.end_point_weight_sum
        if useExp:
            score = np.exp(3 * score)
        return score

    def calcHighLevelTargetScore(self):
        distRobotTarget = np.linalg.norm(self.target - self.robot_pos)
        return -distRobotTarget

    def calcLowLevelTargetScore(self):
        tempTarget = np.array([self.flat_env.robot.walk_target_x, self.flat_env.robot.walk_target_y, 0])
        distRobotTarget = np.linalg.norm(tempTarget - self.robot_pos)
        return -distRobotTarget

    def calcBodyPostureScore(self, useExp=False):
        roll, pitch, yaw = self.flat_env.robot.robot_body.pose().rpy()
        score = -(np.abs(yaw - self.highLevelDegTarget) + np.abs(roll) + np.abs(pitch))
        if useExp:
            score = np.exp(score)
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

    def calcDriftScore(self):
        projection = projPointLineSegment(
            self.robot_pos, self.starting_robot_pos, self.target
        )
        # drawLine(self.robot_pos, projection, [0, 0, 0], lifeTime=0.1, width=1)
        score = np.linalg.norm(projection - self.robot_pos)
        return np.exp(-6 * score)

    def checkTarget(self):
        distToTarget = np.linalg.norm(self.robot_pos - self.target)

        if distToTarget <= 0.5:
            _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()
            randomTarget = self.getRandomVec(self.targetLen, 0, initYaw=yaw)
            newTarget = self.robot_pos + randomTarget
            if self.usePredefinedTarget:
                self.predefinedTargetIndex = (self.predefinedTargetIndex + 1) % len(
                    self.predefinedTarget
                )
                newTarget = self.predefinedTarget[self.predefinedTargetIndex]
            drawLine(self.target, newTarget, [1, 0, 0], lifeTime=0)
            self.starting_robot_pos = self.target.copy()
            self.target = newTarget
            self.starting_ep_pos = self.robot_pos.copy()
            self.highTargetScore = -np.linalg.norm(
                self.target - self.starting_robot_pos
            )
        
        vRobotTargetEnd = self.target - self.robot_pos
        self.highLevelDegTarget = np.arctan2(vRobotTargetEnd[1], vRobotTargetEnd[0])
        self.setWalkTarget(self.target[0], self.target[1])

    def drawDebugRobotPosLine(self):
        if self.cur_timestep % 10 == 0:
            drawLine(self.last_robotPos, self.robot_pos, [1, 1, 1], lifeTime=0)
            self.last_robotPos = self.robot_pos.copy()

    def updateReward(self, action):
        useExp = True
        jointScore, jointVelScore = self.calcJointPosVelLowScore(useExp=useExp)
        # jointScore = self.calcJointScore(useExp=useExp)
        # jointVelScore = self.calcJointVelScore(useExp=useExp)
        # endPointScore = self.calcEndPointScore(useExp=useExp)
        # lowTargetScore = self.calcLowLevelTargetScore()
        bodyPostureScore = self.calcBodyPostureScore(useExp=useExp)

        # self.delta_deltaJoints = (jointScore - self.deltaJoints_low) / 0.0165
        # self.delta_deltaVelJoints = (jointVelScore - self.deltaVelJoints_low) / 0.0165 * 0.1
        # self.delta_deltaEndPoints = (endPointScore - self.deltaEndPoints) / 0.0165
        # self.delta_lowTargetScore = (
        # (lowTargetScore - self.lowTargetScore) / 0.0165 * 0.1
        # )
        # self.delta_bodyPostureScore = (
        # (bodyPostureScore - self.bodyPostureScore) / 0.0165 * 0.1
        # )

        self.deltaJoints_low = jointScore
        self.deltaVelJoints_low = jointVelScore
        # self.deltaJoints = jointScore
        # self.deltaVelJoints = jointVelScore
        # self.deltaEndPoints = endPointScore
        # self.lowTargetScore = lowTargetScore
        self.bodyPostureScore = bodyPostureScore
        # self.baseReward = base_reward
        self.electricityScore = self.calcElectricityCost(action)
        self.jointLimitScore = self.calcJointLimitCost()
        self.aliveReward = self.calcAliveReward()
        self.cumulative_driftScore += self.calcDriftScore()
        self.cumulative_deltaJoints_low += self.deltaJoints_low
        self.cumulative_deltaVelJoints_low += self.deltaVelJoints_low

        reward = [
            self.delta_deltaJoints,
            self.delta_deltaVelJoints,
            self.electricityScore,
            self.jointLimitScore,
            self.aliveReward,
            self.bodyPostureScore
        ]

        rewardWeight = [1, 0.2, 0.1, 0.2, 0.1, 0.4]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        return totalReward / 2

    def updateRewardHigh(self):
        highTargetScore = self.calcHighLevelTargetScore()
        self.delta_highTargetScore = (highTargetScore - self.highTargetScore) / 0.0165
        self.delta_highTargetScore /= (
            self.step_per_level - self.steps_remaining_at_level
        )
        self.highTargetScore = highTargetScore

        self.driftScore = self.cumulative_driftScore / (
            self.step_per_level - self.steps_remaining_at_level
        )
        self.cumulative_driftScore = 0

        deltaJoints_low = self.cumulative_deltaJoints_low / (
            self.step_per_level - self.steps_remaining_at_level
        )
        self.cumulative_deltaJoints_low = 0

        deltaVelJoints_low = self.cumulative_deltaVelJoints_low / (
            self.step_per_level - self.steps_remaining_at_level
        )
        self.cumulative_deltaVelJoints_low = 0

        useExp = True
        jointScore = self.calcJointScore(useExp=useExp)
        jointVelScore = self.calcJointVelScore(useExp=useExp)
        # endPointScore = self.calcEndPointScore(useExp=useExp)
        bodyPostureScore = self.calcBodyPostureScore(useExp=useExp)

        self.deltaJoints = jointScore
        self.deltaVelJoints = jointVelScore
        # self.deltaEndPoints = endPointScore
        self.bodyPostureScore = bodyPostureScore

        rewardHigh = [
            self.delta_highTargetScore,
            self.driftScore,
            self.deltaJoints,
            self.deltaVelJoints,
            self.bodyPostureScore,
            deltaJoints_low,
            deltaVelJoints_low
        ]
        rewardWeightHigh = [0.3, 0.2, 1, 0.3, 0.2, 0.6, 0.4]
        totalRewardHigh = 0
        for r, w in zip(rewardHigh, rewardWeightHigh):
            totalRewardHigh += r * w
        return totalRewardHigh / 3
        
    def high_level_step(self, action, debug=False):
        # # Map sudut agar berada di sekitar -45 s/d 45 derajat
        # actionDegree = np.rad2deg(np.arctan2(action[1], action[0]))
        # _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()
        # # newDegree = np.interp(actionDegree, [-180, 180], [-90, 90]) + np.rad2deg(yaw)
        # newDegree = actionDegree + np.rad2deg(yaw)
        # self.highLevelDegTarget = np.deg2rad(newDegree)

        # Aktifkan jika ingin mengecek apakah policy dari low level env berhasil di import
        # vRobotTargetEnd = self.target - self.robot_pos
        # self.highLevelDegTarget = np.arctan2(vRobotTargetEnd[1], vRobotTargetEnd[0])
        # self.setWalkTarget(self.target[0], self.target[1])
        ##################################

        # cosTarget, sinTarget = np.cos(self.highLevelDegTarget), np.sin(
        #     self.highLevelDegTarget
        # )
        # walkTarget = np.array([self.flat_env.robot.walk_target_x, self.flat_env.robot.walk_target_y, 0])
        # currentDist = np.linalg.norm(walkTarget - self.robot_pos)
        # if(currentDist <= 0.1):
        #     currentDist = self.targetLen
        #     self.lowTargetScore = -self.targetLen
        # newWalkTarget = self.robot_pos + np.array([cosTarget, sinTarget, 0]) * currentDist
        # self.setWalkTarget(newWalkTarget[0], newWalkTarget[1])

        # Re-calculate starting_ep_pos
        # vRobotTarget = newWalkTarget - self.robot_pos
        # lenSEP = np.linalg.norm(self.starting_ep_pos - self.robot_pos)
        # # drawLine(self.robot_pos, self.starting_ep_pos, [1, 1, 1], lifeTime=100)
        # self.starting_ep_pos = -vRobotTarget / np.linalg.norm(vRobotTarget)
        # self.starting_ep_pos *= lenSEP
        # self.starting_ep_pos += self.robot_pos

        self.jointTarget = np.copy(action[2:])

        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps += 1
        self.incFrame(self.skipFrame)

        obs = {self.low_level_agent_id: self.getLowLevelObs()}
        rew = {self.low_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def checkIfDone(self):
        isAlive = self.aliveReward > 0
        isNearTarget = (
            np.linalg.norm(self.target - self.robot_pos)
            <= np.linalg.norm(self.target - self.starting_robot_pos) + 1
        )
        return not (isAlive and isNearTarget)
        # return False

    def low_level_step(self, action, debug=False):
        self.steps_remaining_at_level -= 1

        # Step in the actual env
        # f_obs, f_rew, f_done, _ = self.flat_env.step(action)
        self.flat_env.robot.apply_action(action)
        self.flat_env.scene.global_step()

        f_obs = self.flat_env.robot.calc_state()

        self.cur_obs = f_obs

        totalReward = self.updateReward(action=action)
        
        # self.incFrame(self.skipFrame)

        self.checkTarget()

        if debug:
            self.drawDebugRobotPosLine()

        rew, obs = dict(), dict()
        # Handle env termination & transitions back to higher level
        done = {"__all__": False}
        f_done = False
        if not debug:
            f_done = self.checkIfDone()
        self.cur_timestep += 1
        if f_done or (self.cur_timestep >= self.max_timestep):
            rewardHigh = self.updateRewardHigh()
            done["__all__"] = True
            rew["high_level_agent"] = rewardHigh
            obs["high_level_agent"] = self.getHighLevelObs()
            obs[self.low_level_agent_id] = self.getLowLevelObs()
            rew[self.low_level_agent_id] = totalReward
        elif self.steps_remaining_at_level <= 0:
            rewardHigh = self.updateRewardHigh()
            # done[self.low_level_agent_id] = True
            rew["high_level_agent"] = rewardHigh
            obs["high_level_agent"] = self.getHighLevelObs()
        else:
            obs = {self.low_level_agent_id: self.getLowLevelObs()}
            rew = {self.low_level_agent_id: totalReward}

        return obs, rew, done, {}