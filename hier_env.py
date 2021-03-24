import gym
import pybullet
import pybullet_envs
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

class HierarchicalHumanoidEnv(MultiAgentEnv):

    metadata = {'render.modes': [
        'human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self):
        self.flat_env = gym.make('HumanoidBulletEnv-v0')
        self.action_space = self.flat_env.action_space
        self.observation_space = self.flat_env.observation_space
        self.joints_df = pd.read_csv('Joints CSV/walk8_1JointPos.csv')
        self.end_point_df = pd.read_csv('Joints CSV/walk8_1JointVecFromHip.csv')

        self.frame = 0
        self.frame_update_cnt = 0
        self.max_frame = len(self.joints_df)

        self.max_delta_j_rad = np.deg2rad(5)

        self.rng = np.random.default_rng()

        self.joint_map = {
            'right_knee': 'rightKnee',
            'right_hip_x': 'rightHipX',
            'right_hip_y': 'rightHipY',
            'right_hip_z': 'rightHipZ',
            'left_knee': 'leftKnee',
            'left_hip_x': 'leftHipX',
            'left_hip_y': 'leftHipY',
            'left_hip_z': 'leftHipZ',
        }

        self.joint_weight = {
            'right_knee': 3,
            'right_hip_x': 1,
            'right_hip_y': 3,
            'right_hip_z': 1,
            'left_knee': 3,
            'left_hip_x': 1,
            'left_hip_y': 3,
            'left_hip_z': 1,
        }
        self.joint_weight_sum = sum(self.joint_weight.values())

        self.end_point_map = {
            'link0_12': 'RightLeg',
            'right_foot': 'RightFoot',
            'link0_19': 'LeftLeg',
            'left_foot': 'LeftFoot',
        }
        
        self.end_point_weight = {
            'link0_12': 2,
            'right_foot': 1,
            'link0_19': 2,
            'left_foot': 1,
        }
        self.end_point_weight_sum = sum(self.end_point_weight.values())

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0

    def close(self):
        self.flat_env.close()

    def render(self, mode='human'):
        return self.flat_env.render(mode)

    def setJointsOrientation(self, idx):
        jointsRef = self.joints_df.iloc[idx]
        self.flat_env.jdict['right_knee'].set_state(jointsRef['rightKnee'], 0)

        self.flat_env.jdict['right_hip_x'].set_state(jointsRef['rightHipX'], 0)
        self.flat_env.jdict['right_hip_y'].set_state(jointsRef['rightHipY'], 0)
        self.flat_env.jdict['right_hip_z'].set_state(jointsRef['rightHipZ'], 0)

        self.flat_env.jdict['left_knee'].set_state(jointsRef['leftKnee'], 0)

        self.flat_env.jdict['left_hip_x'].set_state(jointsRef['leftHipX'], 0)
        self.flat_env.jdict['left_hip_y'].set_state(jointsRef['leftHipY'], 0)
        self.flat_env.jdict['left_hip_z'].set_state(jointsRef['leftHipZ'], 0)

    def incFrame(self, inc):
        self.frame =+ inc
        if(self.frame >= self.max_frame - 1):
            self.frame = 0
            self.frame_update_cnt = 0

    def reset(self):
        # Insialisasi dengan posisi awal random sesuai referensi
        return self.resetFromFrame(self.rng.integers(0, self.max_frame-5))

    def resetFromFrame(self, frame):
        self.flat_env.reset()
        self.flat_env.parts['torso'].reset_position([0, 0, 1.15])

        self.frame = frame
        self.frame_update_cnt = 0
        self.setJointsOrientation(self.frame)

        self.incFrame(1)

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0

        self.cur_obs = self.flat_env.robot.calc_state()

        return self.cur_obs

    def resetNonFrame(self):
        self.cur_obs = self.flat_env.reset()
        # self.flat_env.parts['torso'].reset_position([0, 0, 1.15])

        self.frame = 0
        self.frame_update_cnt = 0

        self.deltaJoints = 0
        self.deltaEndPoints = 0
        self.baseReward = 0
        
        return self.cur_obs

    def step(self, action):
        return self._low_level_step(action)

    def calcJointScore(self):
        deltaJoints = 0
        jointsRef = self.joints_df.iloc[self.frame]

        for jMap in self.joint_map:
            deltaJoints += np.abs(self.flat_env.jdict[jMap].get_position() - \
                jointsRef[self.joint_map[jMap]]) * self.joint_weight[jMap]

        return np.exp(-5 * deltaJoints / self.joint_weight_sum)
    
    def calcEndPointScore(self):
        deltaEndPoint = 0
        endPointRef = self.end_point_df.iloc[self.frame]

        base_pos = self.flat_env.parts['lwaist'].get_position()

        for epMap in self.end_point_map:
            v1 = self.flat_env.parts[epMap].get_position()
            v2 = base_pos + getJointPos(endPointRef, self.end_point_map[epMap])
            deltaVec = v2 - v1
            deltaEndPoint += np.linalg.norm(deltaVec) * self.end_point_weight[epMap]
        
        return 2 * np.exp(-10 * deltaEndPoint / self.end_point_weight_sum)


    def calcJumpReward(self, obs):
        return 0

    def low_level_step(self, action):
        logger.debug("Low level agent step {}".format(action))

        # Step in the actual env
        f_obs, f_rew, f_done, _ = self.flat_env.step(action)

        self.cur_obs = f_obs

        # Calculate low-level agent reward
        self.deltaJoints = self.calcJointScore()  # Rentang 0 - 1
        self.deltaEndPoints = self.calcEndPointScore() # Rentang 0 - 1
        self.baseReward = f_rew
        jumpReward = self.calcJumpReward(f_obs)  # Untuk task lompat

        reward = [self.baseReward, self.deltaJoints, self.deltaEndPoints, jumpReward]
        rewardWeight = [1, 0.1, 1, 0]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        if(self.deltaJoints >= 0.5 and self.deltaEndPoints >= 0.5):
            self.incFrame(1)
        else:
            self.frame_update_cnt += 1
            if(self.frame_update_cnt > 10):
                self.incFrame(1)
                self.frame_update_cnt = 0


        return f_obs, totalReward, f_done, {}