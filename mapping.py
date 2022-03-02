import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from pr2_utils import *
from math import cos, sin


def mapupdate(MAP, poses, lidar_tran, ranges, angles):
    '''
    The function update the map by given the lidar data and poses which is position of highest weight

    Input: MAP: map
           poses: 3 * 1 (shape) the position which contains x, y, angle
           lidar_tran: transform matrix
           ranges: valid lidar data
           angles: corresponding angle
    Return:
            updated map object
    '''
    sub = np.array([[0, 0, 0, 1]])
    sizex, sizey, xmin, ymin, res = MAP['sizex'], MAP['sizey'], MAP['xmin'], MAP['ymin'], MAP['res']

    x, y, w = poses

    start_x, start_y = int(x - xmin) // res, int(y - ymin) // res

    # calculate the body translation matrix which transform body frame to world frame
    bodyPose_r = np.array([[cos(w), -sin(w), 0], [sin(w), cos(w), 0], [0, 0, 1]])
    bostPose_v = np.array([[x, y, 1]]).T
    # body_trans is the transfer matrix which include the R and p
    body_trans = np.r_[np.c_[bodyPose_r, bostPose_v], sub]

    end_x, end_y = ranges * np.cos(angles), ranges * np.sin(angles)

    body_pose = np.ones((4, np.prod(end_x.shape)))

    body_pose[0], body_pose[1], body_pose[2] = end_x, end_y, 1.76416
    # transfer the end points to the world frame
    body_pose = np.dot(body_trans, np.dot(lidar_tran, body_pose))

    end_x, end_y = body_pose[0], body_pose[1]
    # the end point which convert to the body pose and quantanilize
    end_x = np.ceil((end_x - xmin) / res).astype(np.int16)
    end_y = np.ceil((end_y - ymin) / res).astype(np.int16)

    for i in range(np.prod(end_x.shape)):

        x, y = int(end_x[i]), int(end_y[i])
        pointsx, pointsy = bresenham2D(start_x, start_y, x, y)
        valid = np.logical_and.reduce((pointsx >= 0, pointsx < sizex, pointsy >= 0, pointsy < sizey))
        xs, ys = pointsx[valid].astype(np.int16), pointsy[valid].astype(np.int16)
        MAP['map'][xs, ys] -= np.log(4)

        if 0 <= x < sizex and 0 <= y < sizey:
            MAP['map'][x, y] += 2 * np.log(4)
    MAP['map'] = np.clip(MAP['map'], 10 * np.log(1 / 4), 10 * np.log(4))
    return MAP


def particle_update(MAP, particle_pose, particle_w, lidar_trans, ranges, angles):
    '''
    Update weights of particles based on lidar scan and map correlation

    Parameters:
    MAP (dictionary) : map object
    partilcle_pose (numpy.ndarray()) : pose of the particles, shape : 3 * n
    particle_w (numpy.ndarray()) : weight of the particles, shape : 3 * n
    lidar_trans (numpy.ndarray()) : transformation matrix which transform the lidar frame to the body frame
    ranges (numpy.ndarray()) : valid lidar scan
    angles (numpy.ndarray()) : valid lidar angle

    Return:
    MAP (dictionary) : map object with updated 2D map
    '''

    # load parameters
    map, sizex, sizey, xmin, ymin, xmax, ymax, res = MAP['map'], MAP['sizex'], MAP['sizey'], MAP['xmin'], MAP['ymin'], \
                                                     MAP['xmax'], MAP['ymax'], MAP['res']

    sub = np.ones((np.prod(ranges.shape)))
    trans_sub = np.array([[0, 0, 0, 1]])

    end_x, end_y = ranges * np.cos(angles).T, ranges * np.sin(angles).T

    #     print (np.c_[end_x, end_y, sub, sub].shape)

    # grid
    idx_x, idx_y = np.arange(-4, 5), np.arange(-4, 5)
    map_x, map_y = np.arange(xmin, xmax + res, res), np.arange(ymin, ymax + res, res)
    map_pmf = (np.exp(map) / (1 + np.exp(map)) < 0.5).astype(np.int16)
    # obs_pmf = (map_pmf < 0.5).astype(np.int16)

    # update on lidar scan
    lidar_scan = np.dot(lidar_trans, np.c_[end_x, end_y, sub, sub].T)  # lidar scan in body frame
    corr_res = []

    for idx in range(particle_pose.shape[1]):
        x, y, w = particle_pose[:, idx]

        body_r_mat = np.array([[cos(w), -sin(w), 0], [sin(w), cos(w), 0], [0, 0, 1]])
        body_p_vec = np.array([[x, y, 1]]).T
        body_trans = np.r_[np.c_[body_r_mat, body_p_vec], trans_sub]

        # transform to world frame
        body_pose = np.dot(body_trans, lidar_scan)  # lidar scan in world frame

        vp = body_pose[0: 2, :]

        curr_corr = mapCorrelation(map_pmf, map_x, map_y, vp, idx_x, idx_y)  # shape : 9 * 9
        corr_res.append(np.max(curr_corr))

    # rescale weigths based on the observation likelihood
    corr_res = np.array(corr_res)
    max_corr = max(corr_res)
    ph = np.exp(corr_res - max_corr) / np.sum(np.exp(corr_res - max_corr))
    particle_w = particle_w * ph / np.sum(particle_w * ph)

    return particle_w


def particle_locate(curr_pos, l_v, a_v, t=1):
    '''
    input:  curr_pos: 3 * n float (shape) current position for all n particles
            l_v: float (shape) the recorded linear velocity
            a_v: float (shape) the recorded angular velocity

    output: 3 * n (shape) the predicted current position after adding the error
    '''
    x, y, w = curr_pos
    posx, posy = l_v * t * np.cos(w + a_v * t), l_v * t * np.sin(w + a_v * t)
    posw = a_v * t

    x_re = posx + x + np.random.normal(0, abs(np.max(posx)) / 10, size=np.prod(x.shape))
    y_re = posy + y + np.random.normal(0, abs(np.max(posy)) / 10, size=np.prod(x.shape))
    w_re = posw + w + np.random.normal(0, abs(np.max(posw)) / 10, size=np.prod(x.shape))

    return np.c_[x_re, y_re, w_re].T

