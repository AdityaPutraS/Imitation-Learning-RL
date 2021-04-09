import numpy as np
from scipy.spatial.transform import Rotation as R

def rotFrom2Vec(v1, v2):
    rotAxis= np.cross(v1, v2)
    rotAxis = rotAxis / np.linalg.norm(rotAxis)
    rotAngle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    rotAngle = np.arccos(rotAngle)
    if(rotAxis.all() == 0):
        rotAxis = np.array([1, 0, 0])
        if((v1 == v2).all()):
            rotAngle = 0
        else:
            rotAngle = np.pi
    return R.from_rotvec(rotAxis * rotAngle)

def map_seq(x, startX, endX, startY, endY):
    return startY + (x - startX)/(endX - startX) * (endY - startY)