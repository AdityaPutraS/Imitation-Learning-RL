from low_level_env import LowLevelHumanoidEnv
import pybullet
import matplotlib.pyplot as plt
import numpy as np

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
    frame = 80 - env.skipFrame
    obs = env.resetFromFrame(frame)
    ep = []
    js = []
    jv = []
    for f in range(env.max_frame):
        env.setJointsOrientation(f)
        ep.append(env.calcEndPointScore())
        js.append(env.calcJointScore())
        jv.append(env.calcJointVelScore())
    env.close()

    x = np.arange(0, env.max_frame, 1)
    epPolyCoef = np.polyfit(x, ep, 3)
    jsPolyCoef = np.polyfit(x, js, 5)
    jsPolyString = ''
    for p, c in enumerate(jsPolyCoef):
        jsPolyString += '{}x^{} + '.format(c, 5 - p)
    print('Joint Score Polynomial: ', jsPolyString)

    epPoly = np.poly1d(epPolyCoef)
    jsPoly = np.poly1d(jsPolyCoef)

    plt.plot(ep)
    plt.plot(js)
    plt.plot(jv)
    
    # plt.plot(x, epPoly(x), '--')
    # plt.plot(x, jsPoly(x), '--')

    plt.legend(['End Point', 'Joint Score', 'End Point Approx', 'Joint Score Approx'])
    plt.show()