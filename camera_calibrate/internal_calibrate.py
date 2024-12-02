import os
import cv2
import numpy as np
import glob
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
w = 7
h = 10
i = 0 # 第i张图片


# 准备棋盘格角点的三维坐标
objp = np.zeros((w*h, 3), np.float32)
objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# 存储所有图片的三维点和二维点
objpoints = []  # 真实世界中的点
imgpoints = []  # 图片中的点

# 读取标定图片
images = glob.glob("cali_image/*.bmp")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h))

    # 如果找到足够的角点，添加到点集中
    if ret == True:
        i = i + 1
        print(f'第{i}张图片')
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)


        # 绘制角点并显示
        img = cv2.drawChessboardCorners(img, (w,h), corners, ret)
        # img_resized = cv2.resize(img, (480, 360))
        # cv2.imshow('findCorners', img_resized)
        # cv2.waitKey(500)
        print('findcorners done')
    if ret == False:
        print('no corner')
    if len(corners) != w * h:
        print("角点数量不匹配")
        continue
    if len(imgpoints) != len(objpoints):
        print("Invalid image skipped")
        continue

cv2.destroyAllWindows()

# 相机标定None, None)
print(ret)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix : \n")
print(mtx)
print("Distortion coefficients : \n")
print(dist)
print("Rotation Vectors : \n")
print(rvecs)
print("Translation Vectors : \n")
print(tvecs)
np.savez('params_int/calibrationInt.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)