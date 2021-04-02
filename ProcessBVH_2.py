import pandas as pd
import numpy as np

from pymo.parsers import BVHParser
from pymo.preprocessing import *

from math_util import rotFrom2Vec
from low_level_env import LowLevelHumanoidEnv
import gym
import pybullet_envs
import pybullet
import time

BVH_FILE = 'Dataset/CMU_Mocap_BVH/08/08_01.bvh'
BASE_DATASET_PATH = 'Dataset/CMU_Mocap_BVH'
SUBJECT_LIST = ['08']
# MOTION_LIST = [['01', '02', '03', '06', '08', '09', '10']]
MOTION_LIST = [["03"]]
ENDPOINT_LIST = ['LeftLeg', 'LeftFoot', 'RightLeg', 'RightFoot', 'Head', 'LeftForeArm', 'LeftHand', 'RightForeArm', 'RightHand']
JOINT_MULTIPLIER = 1.0/15
OUT_FOLDER = 'Processed Relative Joints CSV'

def _getJointPos(df, joint, frame, multiplier=JOINT_MULTIPLIER):
    x = df.iloc[frame][joint + '_Xposition']
    z = df.iloc[frame][joint + '_Zposition']
    y = df.iloc[frame][joint + '_Yposition']
    return np.array([z, x, y]) * multiplier

def getJointPos(df, joint, frame):
    xStart, yStart, zStart = _getJointPos(df, 'Hips', 1)
    return _getJointPos(df, joint, frame) - np.array([xStart, yStart, 0.01])

def getVector2Parts(parts, p1, p2):
    return parts[p2].get_position() - parts[p1].get_position()

def getVector2Joint(df, j1, j2, frame):
    return getJointPos(df, j2, frame) - getJointPos(df, j1, frame)

def getRot(parts, l1, l2, df, j1, j2, frame):
    v1 = getVector2Parts(parts, l1, l2)
    v2 = getVector2Joint(df, j1, j2, frame)
    return rotFrom2Vec(v1, v2).as_euler('xyz')

def moveJoint(l1, l2, j1, j2, frame, part_name, weight, index, env, df):
    rot = getRot(env.parts, l1, l2, df, j1, j2, frame)
    cur_pos = [env.jdict[p].get_position() for p in part_name]
    
    pos = []
    for i, idx in enumerate(index):
        joint_pos = cur_pos[i] + rot[idx] * weight[i]
        env.jdict[part_name[i]].set_state(joint_pos, 0)
        pos.append(joint_pos)
    return pos

def setJoint(frame, env, df):
    rightHipX, rightHipY, rightHipZ = moveJoint('link0_7', 'link0_11', 'RightUpLeg', 'RightLeg', frame, ['right_hip_x', 'right_hip_y', 'right_hip_z'], [1, 1, 1], [0, 1, 2], env, df)
    rightKnee = moveJoint('link0_11', 'right_foot', 'RightLeg', 'RightFoot', frame, ['right_knee'], [-1], [1], env, df)[0]

    leftHipX, leftHipY,leftHipZ = moveJoint('link0_14', 'link0_18', 'LeftUpLeg', 'LeftLeg', frame, ['left_hip_x', 'left_hip_y', 'left_hip_z'], [-1, 1, -1], [0, 1, 2], env, df)
    leftKnee = moveJoint('link0_18', 'left_foot', 'LeftLeg', 'LeftFoot', frame, ['left_knee'], [-1], [1], env, df)[0]
    return [rightHipX, rightHipY, rightHipZ, rightKnee, leftHipX, leftHipY, leftHipZ, leftKnee]

if(__name__ == '__main__'):
    env = gym.make('HumanoidBulletEnv-v0')
    for idxSub, sub in enumerate(SUBJECT_LIST):
        for mot in MOTION_LIST[idxSub]:
            print('Processing Motion {}_{}'.format(sub, mot))
            parser = BVHParser()
            parsed_data = parser.parse('{}/{}/{}_{}.bvh'.format(BASE_DATASET_PATH, sub, sub, mot))
            mp = MocapParameterizer('position')
            bvh_pos = mp.fit_transform([parsed_data])[0].values

            # Process vektor hips -> joint untuk setiap endpoint
            normalized_data = np.zeros((1, len(ENDPOINT_LIST) * 3))
            for i in range(1, len(bvh_pos)):
                basePos = _getJointPos(bvh_pos, 'Hips', i, JOINT_MULTIPLIER)
                tmp = []
                for ep in ENDPOINT_LIST:
                    epPos = _getJointPos(bvh_pos, ep, i, JOINT_MULTIPLIER)
                    tmp = np.hstack((tmp, epPos - basePos))
                normalized_data = np.vstack((normalized_data, tmp))
            normalized_data = normalized_data[1:]
            norm_df = pd.DataFrame(normalized_data, columns=['{}_{}position'.format(ep, axis) for ep in ENDPOINT_LIST for axis in ['X', 'Y', 'Z']])
            norm_df.to_csv('{}/walk{}_{}JointVecFromHip.csv'.format(OUT_FOLDER, sub, mot), index=False)
            print('    Done process normalized vector for motion {}_{}'.format(sub, mot))
            print('    Calculate joint pos for motion {}_{}'.format(sub, mot))

            env.render()
            obs = env.reset()
            joint_data = []
            for i in range(1, len(bvh_pos)):
                joint_data.append(setJoint(i, env, bvh_pos))
                time.sleep(1.0/60)
            joint_df = pd.DataFrame(joint_data, columns=['rightHipX', 'rightHipY', 'rightHipZ', 'rightKnee', 'leftHipX', 'leftHipY', 'leftHipZ', 'leftKnee'])
            joint_df.to_csv('{}/walk{}_{}JointPosRad.csv'.format(OUT_FOLDER, sub, mot), index=False)
            print('    Done calculate joint pos for motion {}_{}'.format(sub, mot))
    env.close()
            


