from pybullet_envs.robot_locomotors import Humanoid
from pybullet_envs.scene_stadium import StadiumScene
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv
import gym

class CustomHumanoid(HumanoidBulletEnv):

    def __init__(self, render=False):
        self.robot = Humanoid()
        HumanoidBulletEnv.__init__(self, self.robot, render)

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = StadiumScene(bullet_client,
                                          gravity=9.8,
                                          timestep=0.0165 / 4,
                                          frame_skip=4)
        return self.stadium_scene