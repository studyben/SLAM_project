import numpy as np
import pr2_utils
import matplotlib.pyplot as plt
import time
from math import cos, sin
from time import sleep
from pr2_utils import *
from initialize import *
from matplotlib.pyplot import figure

from mapping import *

if __name__ == '__main__':
    # Load parameters
    fog_trans = fog_trans()
    lidar_trans = lidar_trans()
    lcam_trans = cam_trans()
    # encoder param
    encoder_res = 4096
    d_l, d_r, w_b = 0.623479, 0.622806, 1.52439
    # stereo param
    baseline = 475.143600050775
    # lidar param
    encoder_fname, fog_fname, lidar_fname = './sensor_data/encoder.csv', './sensor_data/fog.csv', './sensor_data/lidar.csv'
    timeF, fog = pr2_utils.read_data_from_csv(fog_fname)  # fog
    timeE, encoder = pr2_utils.read_data_from_csv(encoder_fname)  # encoder
    timeL, lidar = pr2_utils.read_data_from_csv(lidar_fname)  # lidar
    MAP = map_init()
    angles = np.linspace(-5, 180, 286) / 180 * np.pi
    scan = lidar[0, :]
    valid = np.logical_and((scan < 70), (scan > 2))
    angles, scan = angles[valid], scan[valid]

    initial_pose = np.zeros((3, 1))

    # from update_map import update_map
    MAP = mapupdate(MAP, initial_pose, lidar_trans, scan, angles)

    len_fog = np.prod(timeF.shape)
    num_particles = 10
    particle = np.zeros((3, num_particles))
    particle_w = 1 / num_particles * np.ones((1, num_particles))
    robot_pose_storage = np.zeros((2, 1))

    for i in range(len_fog):
        # load the angular v and linear v every 10 iterations
        if i % 10 == 0:
            #find out nearest time encoder data
            encoder_idx = np.argmin(abs(timeE - timeF[i]))
            if encoder_idx != 0:
                lin_v = sum(encoder[encoder_idx, :] - encoder[encoder_idx - 1, :]) / 2 * np.pi * (d_r + d_l) / 2 / encoder_res
                ang_v = fog[i, 2]
            else:
                lin_v = 0
                ang_v = fog[i, 2]

        # update particles and map every 100 iterations
        if i % 100 == 0:
            # update the particles by using current location and some random noise
            particle = particle_locate(particle, lin_v, ang_v)
            # find out the nearest lidar data
            lidar_idx = np.argmin(abs(timeL - timeF[i]))
            scan, angles = lidar[lidar_idx], np.linspace(-5, 180, 286) / 180 * np.pi
            # filtering out the data that is too small or too large
            valid = np.logical_and(scan < 70, scan > 2)
            scan, angles = scan[valid], angles[valid]
            # the probability of particles is calculated
            particle_w = particle_update(MAP, particle, particle_w, lidar_trans, scan, angles)
            # find the particle with the largest probability
            best_pose = particle[:, np.argmax(particle_w)]
            robot_pose_storage = np.hstack((robot_pose_storage, best_pose[0: 2].reshape((2, 1))))
            #update the map based on the pose
            MAP = mapupdate(MAP, best_pose, lidar_trans, scan, angles)

            # resample particles if number of particles gets low
            num_particles_eff = 1 / np.dot(particle_w.reshape(1, num_particles), \
                                           particle_w.reshape(num_particles, 1))
            # when the square of the particle probability is too small, resample it
            if num_particles_eff < num_particles * 0.25:
                particle, particle_w = np.zeros((3, num_particles)), 1 / num_particles * np.ones((1, num_particles))
    # show the map
    map_pmf = (np.exp(MAP['map']) / (1 + np.exp(MAP['map'])) < 0.15).astype(np.int16)
    x, y = robot_pose_storage[0, :], robot_pose_storage[1, :]  # meters

    # convert trajectory from meters to grid
    grid_x = np.ceil((x - MAP['xmin']) / MAP['res']).astype(np.int16)
    grid_y = np.ceil((y - MAP['ymin']) / MAP['res']).astype(np.int16)

    valid_idx = np.logical_and.reduce((grid_x >= 0, grid_x < MAP['sizex'], grid_y >= 0, grid_y < MAP['sizey']))
    map_pmf[grid_x[valid_idx], grid_y[valid_idx]] = 0

    figure(figsize=(10, 8), dpi=80)
    plt.imshow(map_pmf.T, cmap='binary')
    s = [1] * len(grid_x)
    plt.scatter(grid_x, grid_y, s=s, c='r')
    plt.xlabel('Width, m')
    plt.ylabel('Length, m')
    plt.suptitle('Occupency Grid Map ')
    plt.show()
    plt.savefig('occupency.png')
