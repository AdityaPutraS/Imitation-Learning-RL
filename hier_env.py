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

def getSphere(x, y, z):
    body = pybullet.loadURDF(os.path.join(pybullet_data.getDataPath(), "sphere2red_nocol.urdf"), [x, y, z])
    part_name, _ = pybullet.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    return BodyPart(pybullet, part_name, bodies, 0, -1)


class HierarchicalHumanoidEnv(MultiAgentEnv):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        self.flat_env = gym.make("HumanoidBulletEnv-v0")
        self.action_space = self.flat_env.action_space
        self.observation_space = self.flat_env.observation_space

        reference_list = [
            ("Joints CSV/walk8_1JointPos.csv", "Joints CSV/walk8_1JointVecFromHip.csv"),
        ]
        num_reference = len(reference_list)

        self.high_level_obs_space = self.observation_space
        self.high_level_act_space = Box(
            low=-1.0, high=1.0, shape=[2]
        )  # [target_x, target_y]

        self.low_level_obs_space = Box(
            low=-np.inf, high=np.inf, shape=[17*2 + 2]
        )
        self.low_level_act_space = self.action_space

        self.step_per_level = 8
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
            "link0_12": "RightLeg",
            "right_foot": "RightFoot",
            "link0_19": "LeftLeg",
            "left_foot": "LeftFoot",
        }

        self.end_point_weight = {
            "link0_12": 2,
            "right_foot": 1,
            "link0_19": 2,
            "left_foot": 1,
        }
        self.end_point_weight_sum = sum(self.end_point_weight.values())

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0

        self.target = getSphere(1, 0, 0)
        self.targetHighLevel = getSphere(0, 0, 0)
        self.skipFrame = 2

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
        self.frame = +inc
        if self.frame >= self.max_frame - 1:
            self.frame = 0
            self.frame_update_cnt = 0

    def reset(self):
        # Insialisasi dengan posisi awal random sesuai referensi
        return self.resetFromFrame(self.rng.integers(0, self.max_frame - 5))

    def resetFromFrame(self, frame, resetYaw=0):
        self.flat_env.reset()
        self.flat_env.parts["torso"].reset_position([0, 0, 1.15])
        self.flat_env.parts["torso"].reset_orientation(R.from_euler('z', resetYaw, degrees=True).as_quat())

        self.frame = frame
        self.frame_update_cnt = 0
        self.setJointsOrientation(self.frame)

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0

        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0

        randomX = self.rng.integers(-20, 20)
        randomY = self.rng.integers(-20, 20)
        self.target.reset_position([randomX, randomY, 0])
        self.targetHighLevel.reset_position([0, 0, 0])
        self.flat_env.walk_target_x = randomX
        self.flat_env.walk_target_y = randomY
        
        self.skipFrame = 2

        self.incFrame(self.skipFrame)

        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)

        self.cur_obs = self.flat_env.robot.calc_state()
        return {
            "high_level_agent": self.cur_obs,
        }

    def resetNonFrame(self):
        self.cur_obs = self.flat_env.reset()
        # self.flat_env.parts['torso'].reset_position([0, 0, 1.15])

        self.frame = 0
        self.frame_update_cnt = 0

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0

        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps = 0

        self.target.reset_position([0, 0, 0])
        self.targetHighLevel.reset_position([0, 0, 0])
        self.skipFrame = 2

        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
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
        r = R.from_euler('z', np.arctan2(self.targetHighLevel.get_position()[1], self.targetHighLevel.get_position()[0]))

        for epMap in self.end_point_map:
            v1 = self.flat_env.parts[epMap].get_position()
            v2 = base_pos + r.apply(getJointPos(endPointRef, self.end_point_map[epMap]))
            deltaVec = v2 - v1
            deltaEndPoint += np.linalg.norm(deltaVec) * self.end_point_weight[epMap]

        return 2 * np.exp(-10 * deltaEndPoint / self.end_point_weight_sum)
    
    def calcTargetScore(self):
        vTargetHL = self.targetHighLevel.get_position()
        vTarget = self.target.get_position()
        vRobot = self.flat_env.parts['torso'].get_position()

        # Map ke 2d XY
        vTargetHL[2] = 0
        vTarget[2] = 0
        vRobot[2] = 0

        # Hitung jarak
        dRobotTargetHL = np.linalg.norm(vTargetHL - vRobot)
        dTargetHLTarget = np.linalg.norm(vTarget - vTargetHL)

        return np.exp(-(dRobotTargetHL + (dTargetHLTarget / 10)))

    def calcJumpReward(self, obs):
        return 0

    def getLowLevelObs(self):
        # Berikan low level agent nilai joint dan target high level
        jointVal = self.cur_obs[8:8+17*2]

        return {self.low_level_agent_id: np.hstack((jointVal, self.targetHighLevel.get_position()[:2]))}

    def getHighLevelObs(self):
        return {'high_level_agent': self.cur_obs}

    def high_level_step(self, action):
        self.current_goal = action

        self.targetHighLevel.reset_position([action[0], action[1], 0])

        self.steps_remaining_at_level = self.step_per_level
        self.num_high_level_steps += 1
        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)

        obs = self.getLowLevelObs()
        rew = {self.low_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def low_level_step(self, action):
        self.steps_remaining_at_level -= 1

        # Step in the actual env
        f_obs, f_rew, f_done, _ = self.flat_env.step(action)

        self.cur_obs = f_obs

        # Calculate low-level agent reward
        self.deltaJoints = self.calcJointScore()  # Rentang 0 - 1
        self.deltaEndPoints = self.calcEndPointScore()  # Rentang 0 - 1
        self.baseReward = f_rew
        jumpReward = self.calcJumpReward(f_obs)  # Untuk task lompat

        reward = [self.baseReward, self.deltaJoints, self.deltaEndPoints, jumpReward]
        rewardWeight = [1, 1, 1, 0]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        if self.deltaJoints >= 0.5 and self.deltaEndPoints >= 0.5:
            self.incFrame(self.skipFrame)
        else:
            self.frame_update_cnt += 1
            if self.frame_update_cnt > 10:
                self.incFrame(self.skipFrame)
                self.frame_update_cnt = 0

        # obs = {self.low_level_agent_id: np.hstack((f_obs, self.current_goal))}
        obs = self.getLowLevelObs()
        rew = {self.low_level_agent_id: totalReward}

        # Handle env termination & transitions back to higher level
        done = {"__all__": False}
        if f_done:
            done["__all__"] = True
            rew["high_level_agent"] = self.calcTargetScore()
            obs["high_level_agent"] = self.getHighLevelObs()
        elif self.steps_remaining_at_level <= 0:
            done[self.low_level_agent_id] = True
            rew["high_level_agent"] = self.calcTargetScore()
            obs["high_level_agent"] = self.getHighLevelObs()

        return obs, rew, done, {}