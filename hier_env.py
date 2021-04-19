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


class HierarchicalHumanoidEnv(MultiAgentEnv):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        self.flat_env = HumanoidBulletEnv()
        # self.action_space = self.flat_env.action_space
        # self.observation_space = self.flat_env.observation_space

        self.motion_list = ["motion08_03", "motion09_03"]

        self.high_level_obs_space = self.flat_env.observation_space
        self.high_level_act_space = Box(
            low=-1, high=1, shape=[2]
        )  # cos(target), sin(target)

        self.low_level_obs_space = Box(low=-np.inf, high=np.inf, shape=[42 + 8 * 2])
        self.low_level_act_space = self.flat_env.action_space

        self.step_per_level = 10
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
        self.highLevelDegTarget = 0 # Sudut yang harus dicapai oleh robot (radian)

        self.skipFrame = 1

        # Menandakan motion apa yang harus dijalankan sekarang dan pada frame berapa
        self.selected_motion = 1
        self.selected_motion_frame = 0

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
        self.highTargetScore = -self.targetLen
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

        if self.selected_motion_frame == 0:
            self.starting_ep_pos = self.robot_pos.copy()

    def reset(self):
        # Insialisasi dengan posisi awal random sesuai referensi
        return self.resetFromFrame(
            startFrame=self.rng.integers(0, self.max_frame[self.selected_motion] - 5),
            resetYaw=0,
        )

    def setWalkTarget(self, x, y):
        self.flat_env.walk_target_x = x
        self.flat_env.walk_target_y = y
        self.flat_env.robot.walk_target_x = x
        self.flat_env.robot.walk_target_y = y

    def getRandomVec(self, vecLen, z, initYaw=0):
        # randomRad = initYaw + np.deg2rad(self.rng.integers(-180, 180))
        randomRad = initYaw + np.deg2rad(self.rng.integers(-45, 45))
        randomX = np.cos(randomRad) * vecLen
        randomY = np.sin(randomRad) * vecLen

        return np.array([randomX, randomY, z])

    def resetFromFrame(self, startFrame=0, resetYaw=0):
        self.flat_env.reset()

        self.cur_timestep = 0
        
        # Init target
        self.target = self.getRandomVec(self.targetLen, 0)
        self.highLevelDegTarget = 0
        self.setWalkTarget(self.target[0], self.target[1])

        self.selected_motion = 1  # 0 = 08_03, 1 = 09_03
        self.selected_motion_frame = startFrame
        self.setJointsOrientation(self.selected_motion, self.selected_motion_frame)

        # Posisi awal robot
        # robotPos = self.getRandomVec(3, 1.15)
        robotPos = np.array([0, 0, 1.15])
        self.robot_pos = np.array([robotPos[0], robotPos[1], 0])
        self.last_robotPos  = self.robot_pos.copy()
        self.flat_env.robot.robot_body.reset_position(robotPos)
        
        degToTarget = np.rad2deg(np.arctan2(self.target[1], self.target[0]))
        robotRot = R.from_euler("z", degToTarget + resetYaw, degrees=True)
        self.flat_env.robot.robot_body.reset_orientation(robotRot.as_quat())

        endPointRef = self.end_point_df[self.selected_motion].iloc[self.selected_motion_frame]
        endPointRefNext = self.end_point_df[self.selected_motion].iloc[self.selected_motion_frame + 1]

        # Gunakan kaki kanan sebagai acuan
        rightFootPosActual = self.flat_env.parts["right_foot"].get_position()
        rightFootPosActual[2] = 0
        rotDeg = R.from_euler("z", degToTarget, degrees=True)
        rightFootPosRef = rotDeg.apply(getJointPos(endPointRef, self.end_point_map["right_foot"]))
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

        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0

        self.frame_update_cnt = 0
        self.incFrame(self.skipFrame)

        # self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        self.low_level_agent_id = "low_level_agent"

        self.cur_obs = self.flat_env.robot.calc_state()
        return {"high_level_agent": self.getHighLevelObs()}

    def getLowLevelObs(self):
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
        return np.hstack((self.cur_obs[:-2], jointTargetObs))

    def getHighLevelObs(self):
        targetTheta = np.arctan2(
            self.target[1] - self.robot_pos[1], self.target[0] - self.robot_pos[0]
        )
        _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()
        angleToTarget = targetTheta - yaw
        degTarget = [np.cos(angleToTarget), np.sin(angleToTarget)]

        return np.hstack((self.cur_obs[:1], degTarget, self.cur_obs[3:]))

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

        return -deltaJoints / self.joint_weight_sum

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

        return -deltaVel / self.joint_vel_weight_sum

    def calcEndPointScore(self):
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

        return -deltaEndPoint / self.end_point_weight_sum

    def calcHighLevelTargetScore(self):
        distRobotTarget = np.linalg.norm(self.target - self.robot_pos)
        return -distRobotTarget

    def calcLowLevelTargetScore(self):
        _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()
        return -np.abs(yaw - self.highLevelDegTarget)

    def calcJumpReward(self, obs):
        return 0

    def calcAliveReward(self):
        # Didapat dari perhitungan reward alive env humanoid
        z = self.cur_obs[0] + self.flat_env.robot.initial_z
        # return +2 if z > 0.78 else -1
        return +2 if z > 0.75 else -1
    
    def calcElectricityCost(self, action):
        runningCost = -1.0 * float(np.abs(action * self.flat_env.robot.joint_speeds).mean())
        stallCost = -0.1 * float(np.square(action).mean())
        return runningCost + stallCost

    def calcJointLimitCost(self):
        return -0.1 * self.flat_env.robot.joints_at_limit
    

    def checkTarget(self):
        distToTarget = np.linalg.norm(self.robot_pos - self.target)

        if(distToTarget <= 1):
            _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()
            randomTarget = self.getRandomVec(self.targetLen, 0, initYaw=yaw)
            newTarget = self.robot_pos + randomTarget
            drawLine(self.target, newTarget, [1, 0, 0], lifeTime=0)
            self.target = newTarget
            self.starting_ep_pos = self.robot_pos.copy()
            self.highTargetScore = -self.targetLen

    def drawDebugRobotPosLine(self):
        if(self.cur_timestep % 10 == 0):
            drawLine(self.last_robotPos, self.robot_pos, [1, 1, 1], lifeTime=0)
            self.last_robotPos = self.robot_pos.copy()

    def updateReward(self, action):
        jointScore = self.calcJointScore()
        jointVelScore = self.calcJointVelScore()
        endPointScore = self.calcEndPointScore()
        lowTargetScore = self.calcLowLevelTargetScore()
        highTargetScore = self.calcHighLevelTargetScore()

        self.delta_deltaJoints = (jointScore - self.deltaJoints) / 0.0165
        self.delta_deltaVelJoints = (jointVelScore - self.deltaVelJoints) / 0.0165 * 0.1
        self.delta_deltaEndPoints = (endPointScore - self.deltaEndPoints) / 0.0165
        self.delta_lowTargetScore = (lowTargetScore - self.lowTargetScore) / 0.0165 * 0.1
        self.delta_highTargetScore = (highTargetScore - self.highTargetScore) / 0.0165 * 0.1

        self.deltaJoints =  jointScore
        self.deltaVelJoints = jointVelScore
        self.deltaEndPoints = endPointScore
        self.lowTargetScore = lowTargetScore
        self.highTargetScore = highTargetScore
        # self.baseReward = base_reward
        self.electricityScore = self.calcElectricityCost(action)
        self.jointLimitScore = self.calcJointLimitCost()
        self.aliveReward = self.calcAliveReward()

    def high_level_step(self, action):
        # Map sudut agar berada di sekitar -45 s/d 45 derajat
        actionDegree = np.rad2deg(np.arctan2(action[1], action[0]))
        _, _, yaw = self.flat_env.robot.robot_body.pose().rpy()
        newDegree = np.interp(actionDegree, [-180, 180], [-45, 45]) + np.rad2deg(yaw)
        cosTarget, sinTarget = np.cos(np.deg2rad(newDegree)), np.sin(
            np.deg2rad(newDegree)
        )
        self.highLevelDegTarget = np.deg2rad(newDegree)
        newWalkTarget = self.robot_pos + np.array([cosTarget, sinTarget, 0]) * 5
        self.setWalkTarget(newWalkTarget[0], newWalkTarget[1])
        # self.selected_motion = int(np.interp(action[2], [-1, 1], [0, len(self.motion_list) - 1]))
        # self.selected_motion_frame = int(np.interp(action[3], [-1, 1], [0, self.max_frame[self.selected_motion] - 1]))
        # print("Mot, frame: ", self.selected_motion, self.selected_motion_frame)

        self.steps_remaining_at_level = self.step_per_level
        # self.steps_remaining_at_level = self.max_frame[self.selected_motion] - 1
        self.num_high_level_steps += 1
        # self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)

        obs = {self.low_level_agent_id: self.getLowLevelObs()}
        rew = {self.low_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def low_level_step(self, action):
        self.steps_remaining_at_level -= 1

        # Step in the actual env
        # f_obs, f_rew, f_done, _ = self.flat_env.step(action)
        self.flat_env.robot.apply_action(action)
        self.flat_env.scene.global_step()

        f_obs = self.flat_env.robot.calc_state()

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
        ]

        rewardWeight = [0.1, 0.1, 0.5, 0.2, 0.2, 0.2, 0.1]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        self.incFrame(self.skipFrame)

        self.checkTarget()

        # self.drawDebugRobotPosLine()

        rew, obs = dict(), dict()
        # Handle env termination & transitions back to higher level
        done = {"__all__": False}
        f_done = self.aliveReward < 0
        self.cur_timestep += 1
        if f_done or (self.cur_timestep >= self.max_timestep):
            done["__all__"] = True
            rew["high_level_agent"] = self.delta_highTargetScore + self.aliveReward
            obs["high_level_agent"] = self.getHighLevelObs()
            obs[self.low_level_agent_id] = self.getLowLevelObs()
            rew[self.low_level_agent_id] = totalReward
        elif self.steps_remaining_at_level <= 0:
            # done[self.low_level_agent_id] = True
            rew["high_level_agent"] = self.delta_highTargetScore + self.aliveReward
            obs["high_level_agent"] = self.getHighLevelObs()
        else:
            obs = {self.low_level_agent_id: self.getLowLevelObs()}
            rew = {self.low_level_agent_id: totalReward}

        return obs, rew, done, {}