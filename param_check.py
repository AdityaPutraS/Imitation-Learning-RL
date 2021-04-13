from low_level_env import LowLevelHumanoidEnv
import pybullet
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

qKey = ord("q")
rKey = ord("r")


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
    endPointData = []
    jointScoreData = []
    # jointVecData = []
    for frame in range(0, env.max_frame):
        obs = env.resetFromFrame(frame)
        ep = []
        js = []
        # jv = []
        for f in range(env.max_frame):
            env.setJointsOrientation(f)
            ep.append(env.calcEndPointScore(debug=True))
            js.append(env.calcJointScore())
            # jv.append(env.calcJointVelScore())
        endPointData.append(ep)
        jointScoreData.append(js)
        # jointVecData.append(jv)
    env.close()

    endPointData = np.array(endPointData)
    jointScoreData = np.array(jointScoreData)
    # jointVecData = np.array(jointVecData)

    # x = np.arange(0, env.max_frame, 1)
    # epPolyCoef = np.polyfit(x, ep, 3)
    # jsPolyCoef = np.polyfit(x, js, 5)
    # jsPolyString = ''
    # for p, c in enumerate(jsPolyCoef):
    #     jsPolyString += '{}x^{} + '.format(c, 5 - p)
    # print('Joint Score Polynomial: ', jsPolyString)

    # epPoly = np.poly1d(epPolyCoef)
    # jsPoly = np.poly1d(jsPolyCoef)

    # plt.plot(ep)
    # plt.plot(np.exp(-ep) * 3 - 1.8)
    # plt.plot(js)
    # plt.plot(jv)

    # plt.plot(x, epPoly(x), '--')
    # plt.plot(x, jsPoly(x), '--')
    # plt.legend(['End Point Score', 'Joint Score', 'Velocity Score'])
    # plt.legend(['End Point', 'Joint Score', 'End Point Approx', 'Joint Score Approx'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Frame")
    (lineRK,) = plt.plot(endPointData[0, :, 0])
    (lineRF,) = plt.plot(endPointData[0, :, 1])
    (lineLK,) = plt.plot(endPointData[0, :, 2])
    (lineLF,) = plt.plot(endPointData[0, :, 3])
    (lineJS,) = plt.plot(jointScoreData[0])

    def calcScore(data):
        rk = data[:, :, 0]
        rf = data[:, :, 1]
        lk = data[:, :, 2]
        lf = data[:, :, 3]
        return 2* np.exp(-10*(rk + rf + lk + lf)/env.end_point_weight_sum) - 0.5

    scoreData = calcScore(endPointData)
    (score,) = plt.plot(scoreData[0, :])

    plt.ylim(
        [
            min(np.min(endPointData[0, :, :]), np.min(scoreData[0, :])),
            max(np.max(endPointData[0, :, :]), np.max(scoreData[0, :])),
        ]
    )
    plt.legend(list(env.end_point_map.keys()) + ["End Point Score"] + ["Joint Score"])

    plt.subplots_adjust(left=0.25, bottom=0.25)
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=axframe,
        label="Selected Frame",
        valmin=0,
        valmax=env.max_frame - 1,
        valinit=0,
        valstep=1,
    )

    axymin = plt.axes([0.1, 0.25, 0.0225, 0.63])
    ymin_slider = Slider(
        ax=axymin,
        label="Y Min",
        valmin=-7,
        valmax=7,
        valinit=min(np.min(endPointData[0, :, :]), np.min(scoreData[0, :])),
        orientation="vertical",
    )

    axymax = plt.axes([0.15, 0.25, 0.0225, 0.63])
    ymax_slider = Slider(
        ax=axymax,
        label="Y Max",
        valmin=-7,
        valmax=7,
        valinit=max(np.max(endPointData[0, :, :]), np.max(scoreData[0, :])),
        orientation="vertical",
    )

    def update(val):
        lineRK.set_ydata(endPointData[frame_slider.val, :, 0])
        lineRF.set_ydata(endPointData[frame_slider.val, :, 1])
        lineLK.set_ydata(endPointData[frame_slider.val, :, 2])
        lineLF.set_ydata(endPointData[frame_slider.val, :, 3])
        score.set_ydata(scoreData[frame_slider.val, :])
        lineJS.set_ydata(jointScoreData[frame_slider.val])
        fig.canvas.draw_idle()

    def updateAxis(val):
        plt.subplot(111)
        plt.ylim([ymin_slider.val, ymax_slider.val])

        fig.canvas.draw_idle()

    frame_slider.on_changed(update)
    ymin_slider.on_changed(updateAxis)
    ymax_slider.on_changed(updateAxis)

    plt.show()