import cv2
import numpy as np
import glob
import json
import mediapipe as mp

mp_pose = mp.solutions.pose


def calibrate_camera_from_chessboard(images, chessboard_size=(9,6), square_size=0.025):
    """
    使用棋盘格图像标定单个相机，返回K, R, T。
    chessboard_size: (width_points, height_points)
    square_size: 棋盘格每个方格的实际物理大小（米或其它单位）
    """
    # 准备棋盘格的3D参考坐标系点（Z=0平面）
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
    objp = objp * square_size

    objpoints = []
    imgpoints = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # 标定相机
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # 返回第一张图的外参为例(真实使用中应使用特定视角或者平均)
    R, _ = cv2.Rodrigues(rvecs[0])
    T = tvecs[0].reshape(3)

    return K, dist, R, T


def get_config(file_path):
    """从JSON或其他配置文件中读取K,R,T等参数。"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    K = np.array(data['K'])
    R = np.array(data['R'])
    T = np.array(data['T'])
    return K, R, T

def get_depth(depth_path):
    """
    读取深度图，返回深度图数据。
    假设深度图为16位或32位单通道图像（如.exr或.png）
    """
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # depth是H x W的单通道图像，每个像素为深度值(米或毫米)
    return depth

def denormalize_points(normalized_points, image_width, image_height):
    """
    将MediaPipe关键点归一化坐标(范围0~1)转换为像素坐标。
    normalized_points: [(x_norm, y_norm), ...], x_norm,y_norm ∈ [0,1]
    返回像素坐标列表: [(x_pixel, y_pixel), ...]
    """
    pixel_points = []
    for x_norm, y_norm in normalized_points:
        x_pix = x_norm * image_width
        y_pix = y_norm * image_height
        pixel_points.append((x_pix, y_pix))
    return pixel_points

def pixel_to_camera(K, pixel_coords, depth):
    """
    像素坐标系 -> 相机坐标系
    pixel_coords: (x, y) 像素坐标
    depth: 该像素点对应的深度值(与相机坐标系同单位)
    """
    x_p, y_p = pixel_coords
    u = np.array([x_p, y_p, 1.0])
    camera_coords = depth * np.linalg.inv(K).dot(u)
    return camera_coords

def camera_to_world(R, T, camera_coords):
    """
    相机坐标系 -> 世界坐标系
    R: 相机旋转矩阵
    T: 相机平移向量
    camera_coords: (x_c, y_c, z_c)
    世界坐标 = R^-1 * (camera_coords - T)
    """
    world_coords = np.linalg.inv(R).dot(camera_coords - T)
    return world_coords

def triangulate_points(K1, R1, T1, K2, R2, T2, pts1, pts2):
    """
    简单示例：给定两个相机的内外参，以及两幅图像上的特征点匹配对(pts1, pts2)
    使用标准针孔模型进行三角测量求得3D点坐标。

    pts1, pts2: Nx2的像素坐标数组
    返回 3D点坐标 (N x 3)
    """
    # 构造投影矩阵 P = K [R|T]
    P1 = K1 @ np.hstack((R1, T1.reshape(3,1)))
    P2 = K2 @ np.hstack((R2, T2.reshape(3,1)))

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d / pts4d[3])[:3].T  # 转换为 N x 3
    return pts3d

def detect_keypoints_with_mediapipe(image):
    """
    使用MediaPipe检测人体关键点（以pose为例），返回归一化坐标列表。
    """
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return []
        normalized_points = []
        for lm in results.pose_landmarks.landmark:
            normalized_points.append((lm.x, lm.y))  # 归一化坐标
        return normalized_points


if __name__ == '__main__':
    # 示例：标定相机（也可以从配置文件中读取）
    # 使用棋盘格图像标定
    chessboard_imgs = glob.glob('calibration_images/*.jpg')
    K, dist, R, T = calibrate_camera_from_chessboard(chessboard_imgs, chessboard_size=(9,6), square_size=0.025)
    # 或者从配置文件中读取
    # K, R, T = get_config('camera_config.json')

    # 读取RGB图像和深度图
    image_path = 'test_image.jpg'
    depth_path = 'test_depth.png'
    image = cv2.imread(image_path)
    depth = get_depth(depth_path)
    h, w, _ = image.shape

    # 使用MediaPipe检测关键点（pose为例）
    normalized_points = detect_keypoints_with_mediapipe(image)
    # 将归一化坐标转为像素坐标
    pixel_points = denormalize_points(normalized_points, w, h)

    # 遍历关键点，将其从像素坐标与深度图得到3D坐标
    world_points = []
    for (x_pix, y_pix) in pixel_points:
        # 确保像素坐标在图像范围内
        x_int = int(round(x_pix))
        y_int = int(round(y_pix))
        if 0 <= x_int < w and 0 <= y_int < h:
            depth_val = depth[y_int, x_int]  # 从深度图取对应像素的深度
            if depth_val > 0:  # 确保深度有效
                camera_coords = pixel_to_camera(K, (x_pix, y_pix), depth_val)
                world_coord = camera_to_world(R, T, camera_coords)
                world_points.append(world_coord)
            else:
                world_points.append(None)
        else:
            world_points.append(None)

    # 对于多相机标定和点位合并：
    # 假设有第二台相机的K2,R2,T2以及在第二张图上检测到的对应关键点pixel_points_2
    # 可通过triangulate_points进行3D点重建
    # pts3d = triangulate_points(K, R, T, K2, R2, T2, np.array(pixel_points), np.array(pixel_points_2))

    # 此处只是示意，实际需要根据场景和要求进行进一步完善和异常处理。
    print("World points:", world_points)
