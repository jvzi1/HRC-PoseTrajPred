import numpy as np


# 像素坐标系转换相机坐标系
def pixel_to_camera(K, pixel_coords, depth):
    x_p, y_p = pixel_coords
    u = np.array([x_p, y_p, 1.0])
    camera_coords = depth * np.linalg.inv(K).dot(u)
    return camera_coords

# 相机坐标系转换世界坐标系

def camera_to_world(R, t, camera_coords):
    world_coords = np.linalg.inv(R).dot(camera_coords - T)
    return world_coords

def camera_code():
    pass

if __name__ == '__main__':
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    R = np.array()
    T = np.array([0.0, 0.0, 0.0])
    depth = camera_code()