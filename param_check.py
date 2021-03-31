from low_level_env import LowLevelHumanoidEnv
import pybullet
import matplotlib.pyplot as plt

qKey = ord('q')
rKey = ord('r')

def drawLine(c1, c2, color):
    return pybullet.addUserDebugLine(c1, c2, lineColorRGB=color, lineWidth=5)

if __name__ == "__main__":

    # # Interactive Check
    # env = LowLevelHumanoidEnv()
    # env.render()
    # doneAll = False
    # while(not doneAll):
    #     frame = 70
    #     obs = env.resetFromFrame(frame)
    #     pybullet.removeAllUserDebugItems()
    #     env.setJointsOrientation(71)
    #     print(env.calcEndPointScore(), env.calcJointScore())
    #     # for f in range(env.max_frame):
    #     #     env.setJointsOrientation(f)
    #     #     if(f == frame+env.skipFrame):
    #     #         print('===============>', end='')
    #     #     print(f, env.calcEndPointScore(), env.calcJointScore())
    #     done = False
    #     while(not done):
    #         keys = pybullet.getKeyboardEvents()
    #         if qKey in keys and keys[qKey] & pybullet.KEY_WAS_TRIGGERED:
    #             print('QUIT')
    #             done = True
    #             doneAll = True
    #         if rKey in keys and keys[rKey] & pybullet.KEY_WAS_TRIGGERED:
    #             done = True
    # env.close()

    env = LowLevelHumanoidEnv()
    frame = 80
    obs = env.resetFromFrame(frame)
    ep = []
    js = []
    for f in range(env.max_frame):
        env.setJointsOrientation(f)
        ep.append(env.calcEndPointScore())
        js.append(env.calcJointScore())
    env.close()
    plt.plot(ep)
    plt.plot(js)
    plt.legend(['End Point', 'Joint Score'])
    plt.show()