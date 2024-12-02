import cv2
# 2d imagepoints

# 3d objpoints

# calibrate
def stereo_calibrate(objpoints, imgpoints1, imgpoints2, K1, dist1, K2, dist2, image_size):
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, K1, dist1, K2, dist2, image_size, flags=flags)
    return R, T