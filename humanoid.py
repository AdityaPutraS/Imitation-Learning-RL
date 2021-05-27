from pybullet_envs.robot_locomotors import WalkerBase
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_abstract import Scene
import pybullet_data
import os
import pybullet
import numpy as np
import random

# Didapat dari kode pybullet_env
class CustomHumanoidRobot(WalkerBase):
    self_collision = True
    foot_list = []

    def __init__(self):
        WalkerBase.__init__(
            self,
            "humanoid_symmetric_2.xml",
            "torso",
            action_dim=19,
            obs_dim=44,
            power=0.41,
        )

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder_x", "right_shoulder_y", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder_x", "left_shoulder_y", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["right_ankle_y"]
        self.motor_power += [200]
        self.motor_names += ["left_ankle_y"]
        self.motor_power += [200]
        self.motors = [self.jdict[n] for n in self.motor_names]
        if self.random_yaw:
            yaw = self.np_random.uniform(low=-3.14, high=3.14)
            position = [0, 0, 1.4]
            orientation = [
                0,
                0,
                yaw,
            ]  # just face random direction, but stay straight otherwise
            self.robot_body.reset_position(position)
            self.robot_body.reset_orientation(orientation)
        self.initial_z = 0.8

    random_yaw = False
    random_lean = False

    def apply_action(self, a):
        assert np.isfinite(a).all()
        force_gain = 1
        for i, m, power in zip(range(19), self.motors, self.motor_power):
            m.set_motor_torque(
                float(force_gain * power * self.power * np.clip(a[i], -1, +1))
            )

    def alive_bonus(self, z, pitch):
        return (
            +2 if z > 0.78 else -1
        )  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


class CustomScene(Scene):
    multiplayer = False
    sceneLoaded = False
    terrainShape = -1
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns

    def replaceHeightfieldData(self, newData: list):
        self.heightfieldData = newData.copy()
        self.terrainShape = self._p.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD,
            meshScale=[1, 1, 1],
            heightfieldTextureScaling=256,
            heightfieldData=self.heightfieldData,
            numHeightfieldRows=self.numHeightfieldRows,
            numHeightfieldColumns=self.numHeightfieldColumns,
            replaceHeightfieldIndex = self.terrainShape
        )

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(
            self, bullet_client
        )  # contains cpp_world.clean_everything()

        # Generate tinggi floor
        for j in range(int(self.numHeightfieldColumns / 2)):
            for i in range(int(self.numHeightfieldRows / 2)):
                height = random.uniform(0, 0.05) * 10
                self.heightfieldData[2 * i + 2 * j * self.numHeightfieldRows] = height
                self.heightfieldData[2 * i + 1 + 2 * j * self.numHeightfieldRows] = height
                self.heightfieldData[2 * i + (2 * j + 1) * self.numHeightfieldRows] = height
                self.heightfieldData[
                    2 * i + 1 + (2 * j + 1) * self.numHeightfieldRows
                ] = height

        baseHeight = 0
        for j in [63, 64]:
            for i in [63, 64]:
                self.heightfieldData[2 * i + 2 * j * self.numHeightfieldRows] = baseHeight
                self.heightfieldData[2 * i + 1 + 2 * j * self.numHeightfieldRows] = baseHeight
                self.heightfieldData[2 * i + (2 * j + 1) * self.numHeightfieldRows] = baseHeight
                self.heightfieldData[
                    2 * i + 1 + (2 * j + 1) * self.numHeightfieldRows
                ] = baseHeight

        self.terrainShape = self._p.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD,
            meshScale=[1, 1, 1],
            heightfieldTextureScaling=256,
            heightfieldData=self.heightfieldData,
            numHeightfieldRows=self.numHeightfieldRows,
            numHeightfieldColumns=self.numHeightfieldColumns,
            replaceHeightfieldIndex = self.terrainShape
        )

        if not self.sceneLoaded:
            self.sceneLoaded = True

            self.terrain = self._p.createMultiBody(0, self.terrainShape)
            self._p.resetBasePositionAndOrientation(
                self.terrain, [0, 0, 0.25], [0, 0, 0, 1]
            )

            # terrainShape = self._p.createCollisionShape(shapeType = pybullet.GEOM_HEIGHTFIELD, meshScale=[.1,.1,24],fileName = "./wm_height_out.png")
            # textureId = self._p.loadTexture("./gimp_overlay_out.png")
            # self.terrain  = self._p.createMultiBody(0, terrainShape)
            # self._p.changeVisualShape(self.terrain, -1, textureUniqueId = textureId)

        self._p.changeVisualShape(self.terrain, -1, rgbaColor=[1, 1, 1, 0.8])
        self._p.changeDynamics(
            self.terrain, -1, lateralFriction=0.8, restitution=0.5
        )
        self._p.configureDebugVisualizer(
            pybullet.COV_ENABLE_PLANAR_REFLECTION, self.terrain
        )


class CustomHumanoid(MJCFBaseBulletEnv):
    def __init__(self, render=False):
        self.robot = CustomHumanoidRobot()
        self.camera_x = 0
        self.walk_target_x = 10
        self.walk_target_y = 0
        self.stateId = -1
        HumanoidBulletEnv.__init__(self, self.robot, render)

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = CustomScene(
            bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4
        )
        return self.stadium_scene

    def reset(self):
        if self.stateId >= 0:
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        (
            self.parts,
            self.jdict,
            self.ordered_joints,
            self.robot_body,
        ) = self.robot.addToScene(self._p, self.stadium_scene.terrain)

        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
            # print("saving state self.stateId:",self.stateId)

        return r

    def camera_adjust(self):
        x, y, z = self.robot.body_real_xyz

        self.camera_x = x
        self.camera.move_and_look_at(self.camera_x, y, 1.4, x, y, 1.0)
