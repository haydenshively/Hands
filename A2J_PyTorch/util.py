import numpy as np
import cv2


def generate_anchor_cluster(P_h=None, P_w=None):
    if P_h is None:
        P_h = np.array([2,6,10,14])

    if P_w is None:
        P_w = np.array([2,6,10,14])

    num_anchors = len(P_h) * len(P_h)
    anchors = np.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1

    return anchors


def replicate(anchor_cluster, shape, stride):
    shift_h = np.arange(0, shape[0]) * stride
    shift_w = np.arange(0, shape[1]) * stride

    shift_h, shift_w = np.meshgrid(shift_h, shift_w)
    shifts = np.vstack((shift_h.ravel(), shift_w.ravel())).transpose()

    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 2) shifted anchors
    A = anchor_cluster.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchor_cluster.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 2))

    return all_anchors


def pixel2world(x, fx, fy, ux, uy):
    x[:,:,0] = (x[:,:,0] - ux) * x[:,:,2] / fx
    x[:,:,1] = (x[:,:,1] - uy) * x[:,:,2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:,:,0] = x[:,:,0] * fx / x[:,:,2] + ux
    x[:,:,1] = x[:,:,1] * fy / x[:,:,2] + uy
    return x
