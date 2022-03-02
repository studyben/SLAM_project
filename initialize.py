import numpy as np

def fog_trans():
    sub = np.array([[0, 0, 0, 1]])
    fog_r_mat = np.eye(3)
    fog_p_vec = np.array([[-0.335], [-0.035], [0.78]])
    return np.r_[np.c_[fog_r_mat, fog_p_vec], sub]

def lidar_trans():
    sub = np.array([[0, 0, 0, 1]])
    lidar_r_mat = np.array([[0.00130201, 0.796097, 0.605167], [0.999999, -0.000419027, -0.00160026],
                            [-0.00102038, 0.605169, -0.796097]])
    lidar_p_vec = np.array([[0.8349, -0.0126869, 1.76416]]).T
    return np.r_[np.c_[lidar_r_mat, lidar_p_vec], sub]

def cam_trans():
    sub = np.array([[0, 0, 0, 1]])
    lcam_r_mat = np.array([[-0.00680499, -0.0153215, 0.99985], [-0.999977, 0.000334627, -0.00680066],
                           [-0.000230383, -0.999883, -0.0153234]])
    lcam_p_vec = np.array([[1.64239, 0.247401, 1.58411]]).T
    return np.r_[np.c_[lcam_r_mat, lcam_p_vec], sub]

def map_init():
    MAP = {}
    MAP['res'] = 1  # meters
    MAP['xmin'] = -100  # meters
    MAP['ymin'] = -1100
    MAP['xmax'] = 1300
    MAP['ymax'] = 100
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float32)
    return MAP