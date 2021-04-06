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

BVH_FILE = "Dataset/CMU_Mocap_BVH/08/08_01.bvh"
BASE_DATASET_PATH = "Dataset/CMU_Mocap_BVH"
SUBJECT_LIST = ["02"]
# MOTION_LIST = [['01', '02', '03', '06', '08', '09', '10']]
MOTION_LIST = [["04"]]
START_FRAME = [[1]]
END_FRAME = [[300]]
MOTION = [
    {
        "subject": "02",
        "motion_number": "04",
        "start_frame": 1,
        "end_frame": 300,
    }
]
TIME_STEP_BVH = 0.0083333
TIME_STEP_PYBULLET = (
    0.0165  # Didapat dari TimeStep * FrameSkip di gym_locomotion_envs.py milik pybullet
)
ENDPOINT_LIST = [
    "LeftLeg",
    "LeftFoot",
    "RightLeg",
    "RightFoot",
    "Head",
    "LeftForeArm",
    "LeftHand",
    "RightForeArm",
    "RightHand",
]
JOINT_MULTIPLIER = 1.0 / 15
OUT_FOLDER = "Processed Relative Joints CSV"


def _getJointPos(df, joint, frame, multiplier=JOINT_MULTIPLIER):
    x = df.iloc[frame][joint + "_Xposition"]
    z = df.iloc[frame][joint + "_Zposition"]
    y = df.iloc[frame][joint + "_Yposition"]
    return np.array([z, x, y]) * multiplier


def getJointPos(df, joint, frame):
    xStart, yStart, zStart = _getJointPos(df, "Hips", 1)
    return _getJointPos(df, joint, frame) - np.array([xStart, yStart, 0.01])


def getVector2Parts(parts, p1, p2):
    return parts[p2].get_position() - parts[p1].get_position()


def getVector2Joint(df, j1, j2, frame):
    return getJointPos(df, j2, frame) - getJointPos(df, j1, frame)


def getRot(parts, l1, l2, df, j1, j2, frame):
    v1 = getVector2Parts(parts, l1, l2)
    v2 = getVector2Joint(df, j1, j2, frame)
    return rotFrom2Vec(v1, v2).as_euler("xyz")


def moveJoint(l1, l2, j1, j2, frame, part_name, weight, index, env, df):
    rot = getRot(env.parts, l1, l2, df, j1, j2, frame)
    cur_pos = [env.jdict[p].get_position() for p in part_name]

    pos_relative = []
    pos_absolute = []
    for i, idx in enumerate(index):
        joint_pos = cur_pos[i] + rot[idx] * weight[i]
        env.jdict[part_name[i]].set_state(joint_pos, 0)
        joint_absolute_pos = env.jdict[part_name[i]].get_position()
        pos_absolute.append(joint_absolute_pos)
        joint_relative_pos = env.jdict[part_name[i]].current_relative_position()[0]
        pos_relative.append(joint_relative_pos)
    return pos_absolute, pos_relative


def setJoint(frame, env, df):
    rightHip = moveJoint(
        "link0_7",
        "link0_11",
        "RightUpLeg",
        "RightLeg",
        frame,
        ["right_hip_x", "right_hip_y", "right_hip_z"],
        [1, 1, 1],
        [0, 1, 2],
        env,
        df,
    )
    rightKnee = moveJoint(
        "link0_11",
        "right_foot",
        "RightLeg",
        "RightFoot",
        frame,
        ["right_knee"],
        [-1],
        [1],
        env,
        df,
    )
    rightHipX, rightHipY, rightHipZ = rightHip[0]
    rightHipX_relative, rightHipY_relative, rightHipZ_relative = rightHip[1]
    rightKnee_absolute = rightKnee[0][0]
    rightKnee_relative = rightKnee[1][0]

    leftHip = moveJoint(
        "link0_14",
        "link0_18",
        "LeftUpLeg",
        "LeftLeg",
        frame,
        ["left_hip_x", "left_hip_y", "left_hip_z"],
        [-1, 1, -1],
        [0, 1, 2],
        env,
        df,
    )
    leftKnee = moveJoint(
        "link0_18",
        "left_foot",
        "LeftLeg",
        "LeftFoot",
        frame,
        ["left_knee"],
        [-1],
        [1],
        env,
        df,
    )
    leftHipX, leftHipY, leftHipZ = leftHip[0]
    leftHipX_relative, leftHipY_relative, leftHipZ_relative = leftHip[1]
    leftKnee_absolute = leftKnee[0][0]
    leftKnee_relative = leftKnee[1][0]
    return [
        rightHipX,
        rightHipY,
        rightHipZ,
        rightKnee_absolute,
        leftHipX,
        leftHipY,
        leftHipZ,
        leftKnee_absolute,
    ], [
        rightHipX_relative,
        rightHipY_relative,
        rightHipZ_relative,
        rightKnee_relative,
        leftHipX_relative,
        leftHipY_relative,
        leftHipZ_relative,
        leftKnee_relative,
    ]


if __name__ == "__main__":
    env = gym.make("HumanoidBulletEnv-v0")
    for m in MOTION:
        sub = m["subject"]
        mot = m["motion_number"]
        start_frame = m["start_frame"]
        end_frame = m["end_frame"]

        print("Processing Motion {}_{}".format(sub, mot))
        parser = BVHParser()
        parsed_data = parser.parse(
            "{}/{}/{}_{}.bvh".format(BASE_DATASET_PATH, sub, sub, mot)
        )
        mp = MocapParameterizer("position")
        bvh_pos = mp.fit_transform([parsed_data])[0].values

        # Process vektor hips -> joint untuk setiap endpoint
        normalized_data = np.zeros((1, len(ENDPOINT_LIST) * 3))
        for i in range(start_frame, end_frame):
            basePos = _getJointPos(bvh_pos, "Hips", i, JOINT_MULTIPLIER)
            tmp = []
            for ep in ENDPOINT_LIST:
                epPos = _getJointPos(bvh_pos, ep, i, JOINT_MULTIPLIER)
                tmp = np.hstack((tmp, epPos - basePos))
            normalized_data = np.vstack((normalized_data, tmp))
        normalized_data = normalized_data[1:]
        norm_df = pd.DataFrame(
            normalized_data,
            columns=[
                "{}_{}position".format(ep, axis)
                for ep in ENDPOINT_LIST
                for axis in ["X", "Y", "Z"]
            ],
        )
        norm_df.to_csv(
            "{}/motion{}_{}JointVecFromHip.csv".format(OUT_FOLDER, sub, mot),
            index=False,
        )
        print(
            "    Done process normalized vector for motion {}_{}".format(sub, mot)
        )
        print("    Calculate joint pos for motion {}_{}".format(sub, mot))

        env.render()
        obs = env.reset()

        # Hitung nilai radian joint setiap waktu
        joint_data = []
        joint_data_rel = []
        for i in range(start_frame, end_frame):
            absolute, relative = setJoint(i, env, bvh_pos)
            joint_data.append(absolute)
            joint_data_rel.append(relative)
            time.sleep(1.0 / 60)
        joint_df = pd.DataFrame(
            joint_data,
            columns=[
                "rightHipX",
                "rightHipY",
                "rightHipZ",
                "rightKnee",
                "leftHipX",
                "leftHipY",
                "leftHipZ",
                "leftKnee",
            ],
        )
        joint_df.to_csv(
            "{}/motion{}_{}JointPosRad.csv".format(OUT_FOLDER, sub, mot), index=False
        )

        joint_df_rel = pd.DataFrame(
            joint_data_rel,
            columns=[
                "rightHipX",
                "rightHipY",
                "rightHipZ",
                "rightKnee",
                "leftHipX",
                "leftHipY",
                "leftHipZ",
                "leftKnee",
            ],
        )
        joint_df_rel.to_csv(
            "{}/motion{}_{}JointPosRadRelative.csv".format(OUT_FOLDER, sub, mot), index=False
        )
        print("    Done calculate joint pos for motion {}_{}".format(sub, mot))

        # Hitung kecepatan setiap joint dalam rad/s
        joint_vel = [[0 for _ in joint_df.columns]]
        joint_rad_np = joint_df.to_numpy()
        for i in range(start_frame+1, len(joint_rad_np)):
            posAwal = joint_rad_np[i - 1]
            posAkhir = joint_rad_np[i]
            vel = (posAkhir - posAwal) / TIME_STEP_PYBULLET
            joint_vel.append(vel)
        joint_vel_df = pd.DataFrame(joint_vel, columns=joint_df.columns)
        joint_vel_df.to_csv(
            "{}/motion{}_{}JointSpeedRadSec.csv".format(OUT_FOLDER, sub, mot),
            index=False,
        )
        print("    Done calculate joint velocity for motion {}_{}".format(sub, mot))
    env.close()
