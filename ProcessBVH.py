import pandas as pd
import numpy as np

from pymo.parsers import BVHParser
from pymo.preprocessing import *

from math_util import rotFrom2Vec
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv
from humanoid import CustomHumanoidRobot
import gym
import pybullet_envs
import pybullet
import time

BVH_FILE = "Dataset/CMU_Mocap_BVH/08/08_01.bvh"
BASE_DATASET_PATH = "Dataset/CMU_Mocap_BVH"
MOTION = [
    {
        "subject": "02",
        "motion_number": "04",
        "start_frame": 1,
        "end_frame": 300,
    },
    {
        "subject": "08",
        "motion_number": "03",
        "start_frame": 1,
        "end_frame": 127,
    },
    {
        "subject": "09",
        "motion_number": "03",
        "start_frame": 1,
        "end_frame": 91,
    },
    {
        "subject": "13",
        "motion_number": "13",
        "start_frame": 100,
        "end_frame": 320,
    },
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
OUT_FOLDER = "Joints CSV With Hand"


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

# Link 7,8, 9 = right hip x, y, z
# Link 11 = right knee
# Link 13, 14 = right ankle x, y

# Link 16, 17, 18 = left hip x, y, z
# Link 20 = left knee
# Link 22, 23 = left ankle x, y

# Link 25, 26 = right shoulder y, x
# Link 28 = right elbow

# Link 31, 32 = left shoulder y, x
# Link 34 = left elbow
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
        "link0_16",
        "link0_20",
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
        "link0_20",
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

    rightHand = moveJoint(
        "link0_25",
        "link0_28",
        "RightArm",
        "RightForeArm",
        frame,
        ["right_shoulder_x", "right_shoulder_y"],
        [1, 1],
        [0, 1],
        env,
        df
    )
    rightElbow = moveJoint(
        "link0_28",
        "right_hand",
        "RightForeArm",
        "RightHand",
        frame,
        ["right_elbow"],
        [1],
        [1],
        env,
        df
    )

    rightShoulderX, rightShoulderY = rightHand[0]
    rightShoulderX_relative, rightShoulderY_relative = rightHand[1]
    rightElbow_absolute = rightElbow[0][0]
    rightElbow_relative = rightElbow[1][0]

    leftHand = moveJoint(
        "link0_31",
        "link0_34",
        "LeftArm",
        "LeftForeArm",
        frame,
        ["left_shoulder_x", "left_shoulder_y"],
        [-1, 1],
        [0, 1],
        env,
        df
    )
    leftElbow = moveJoint(
        "link0_34",
        "left_hand",
        "LeftForeArm",
        "LeftHand",
        frame,
        ["left_elbow"],
        [1],
        [1],
        env,
        df
    )

    leftShoulderX, leftShoulderY = leftHand[0]
    leftShoulderX_relative, leftShoulderY_relative = leftHand[1]
    leftElbow_absolute = leftElbow[0][0]
    leftElbow_relative = leftElbow[1][0]

    return [
        rightHipX,
        rightHipY,
        rightHipZ,
        rightKnee_absolute,
        leftHipX,
        leftHipY,
        leftHipZ,
        leftKnee_absolute,
        rightShoulderX,
        rightShoulderY,
        rightElbow_absolute,
        leftShoulderX,
        leftShoulderY,
        leftElbow_absolute,
    ], [
        rightHipX_relative,
        rightHipY_relative,
        rightHipZ_relative,
        rightKnee_relative,
        leftHipX_relative,
        leftHipY_relative,
        leftHipZ_relative,
        leftKnee_relative,
        rightShoulderX_relative,
        rightShoulderY_relative,
        rightElbow_relative,
        leftShoulderX_relative,
        leftShoulderY_relative,
        leftElbow_relative,
    ]


# Jika USE_NORMALIZED_DF bernilai true, dataset jointvecfromhip akan berisi vektor tiap sendi relatif terhadap hips pada setiap framenya
# Jika di visualisasikan sama saja seperti melakukan gerakan tapi posisi robot tetap diam di tempat
# Jika USE_NORMALIZED_DF bernilai false, dataset berisi posisi absolut setiap sendi dalam koordinat global
# Jika di visualisasikan maka akan terlihat berjalan / melompat, percis seperti di program blender
USE_NORMALIZED_DF = False

if __name__ == "__main__":
    env = HumanoidBulletEnv(robot=CustomHumanoidRobot())
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
                if USE_NORMALIZED_DF:
                    epPos = _getJointPos(bvh_pos, ep, i, JOINT_MULTIPLIER)
                    tmp = np.hstack((tmp, epPos - basePos))
                else:
                    epPos = getJointPos(bvh_pos, ep, i)
                    tmp = np.hstack((tmp, epPos))
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
                "rightShoulderX",
                "rightShoulderY",
                "rightElbow",
                "leftShoulderX",
                "leftShoulderY",
                "leftElbow",
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
                "rightShoulderX",
                "rightShoulderY",
                "rightElbow",
                "leftShoulderX",
                "leftShoulderY",
                "leftElbow",
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
