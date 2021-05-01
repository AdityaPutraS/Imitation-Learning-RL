from pybullet_envs.robot_locomotors import WalkerBase
import numpy as np

# Didapat dari kode pybullet_env
class Humanoid(WalkerBase):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    def __init__(self):
        WalkerBase.__init__(self,
                        'humanoid_symmetric_2.xml',
                        'torso',
                        action_dim=17,
                        obs_dim=44,
                        power=0.41)
    # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]
        if self.random_yaw:
            yaw = self.np_random.uniform(low=-3.14, high=3.14)
            position = [0, 0, 1.4]
            orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
            self.robot_body.reset_position(position)
            self.robot_body.reset_orientation(orientation)
        self.initial_z = 0.8

    random_yaw = False
    random_lean = False

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        force_gain = 1
        for i, m, power in zip(range(17), self.motors, self.motor_power):
            m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying