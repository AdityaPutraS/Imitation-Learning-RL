import gym
import pybullet
import pybullet_envs
from gym.spaces import Box, Discrete, Tuple
import logging
import random

import pandas as pd
import numpy as np

from ray.rllib.env import MultiAgentEnv

from scipy.spatial.transform import Rotation as R
from math_util import rotFrom2Vec

logger = logging.getLogger(__name__)


class LowLevelHumanoidEnv(gym.Env):
    metadata = {'render.modes': [
        'human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self):
        self.flat_env = gym.make('HumanoidBulletEnv-v0')
        self.action_space = self.flat_env.action_space
        self.observation_space = self.flat_env.observation_space
        self.joints_df = pd.read_csv('Joints CSV/walk8_1JointPos.csv')

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

    def reset(self):
        # Insialisasi dengan posisi awal random sesuai referensi
        return self.resetFromFrame(self.rng.integers(0, self.max_frame-1))

    def resetFromFrame(self, frame):
        self.flat_env.reset()
        self.flat_env.parts['torso'].reset_position([0, 0, 1.15])

        self.frame = frame
        self.setJointsOrientation(self.frame)

        self.cur_obs = self.flat_env.robot.calc_state()

        return self.cur_obs

    def resetNonFrame(self):
        self.cur_obs = self.flat_env.reset()
        self.flat_env.parts['torso'].reset_position([0, 0, 1.15])

        self.frame = 0
        return self.cur_obs

    def step(self, action):
        return self._low_level_step(action)

    def calcJointScore(self):
        deltaJoints = 0
        jointsRef = self.joints_df.iloc[self.frame+1]

        for jMap in self.joint_map:
            deltaJoints += np.abs(self.flat_env.jdict[jMap].get_position() - \
                jointsRef[self.joint_map[jMap]])

        return np.exp(-10 * deltaJoints / 8)

    def calcJumpReward(self, obs):
        return 0

    def _low_level_step(self, action):
        logger.debug("Low level agent step {}".format(action))

        # Step in the actual env
        f_obs, f_rew, f_done, _ = self.flat_env.step(action)

        self.cur_obs = f_obs

        # Calculate low-level agent reward
        deltaJoints = self.calcJointScore()  # Rentang 0 - 1
        jumpReward = self.calcJumpReward(f_obs)  # Untuk task lompat

        reward = [f_rew, deltaJoints, jumpReward]
        rewardWeight = [0, 1, 0]

        totalReward = 0
        for r, w in zip(reward, rewardWeight):
            totalReward += r * w

        # self.frame_update_cnt += 1
        # if(self.frame_update_cnt >= 2):
        #     self.frame += 1
        #     self.frame_update_cnt = 0
        if(deltaJoints >= 0.5):
            self.frame += 5

        if(self.frame >= self.max_frame - 1):
            self.frame = 0
            self.frame_update_cnt = 0

        return f_obs, totalReward, f_done, {}
