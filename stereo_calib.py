import numpy as np
import cv2
import glob
import os
import yaml
w=9
h=6
class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((w*h, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints_l = [] # 2d points in image plane.
        self.imgpoints_r = [] # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(os.path.join(self.cal_path,"video_l.mp4"),os.path.join(self.cal_path,"video_r.mp4"))

    def read_images(self, video_left, video_right):
        # images_right = glob.glob(cal_path + 'RIGHT/*.BMP')
        # images_left = glob.glob(cal_path + 'LEFT/*.BMP')
        # images_left.sort()
        # images_right.sort()
        cap_left = cv2.VideoCapture(video_left)
        cap_right = cv2.VideoCapture(video_right)
        left_buf, right_buf = [], []
        while True:
            if cap_left.grab():
                flag, frame = cap_left.retrieve()
                left_buf.append(frame)
            if cap_left.get(cv2.CAP_PROP_POS_FRAMES) == cap_left.get(cv2.CAP_PROP_FRAME_COUNT):
                break

        while True:
            if cap_right.grab():
                flag, frame = cap_right.retrieve()
                right_buf.append(frame)
            if cap_right.get(cv2.CAP_PROP_POS_FRAMES) == cap_right.get(cv2.CAP_PROP_FRAME_COUNT):
                break
        
        print(len(left_buf),len(right_buf))
        left_buf = np.array(left_buf)
        right_buf = np.array(right_buf)
        idx = np.random.choice(len(left_buf),size=(32,), replace=False)
        left_buf = left_buf[idx]
        right_buf = right_buf[idx]
        for i, (img_l, img_r) in enumerate(zip(left_buf, right_buf)):
            print(i)
            # if i >10:
            #     break
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (w, h), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (w, h), None)

            # If found, add object points, image points (after refining them)
            

            if ret_l is True and ret_r is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)
                
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)
                self.objpoints.append(self.objp)


            img_shape = gray_l.shape[::-1]
        print("mono calibrate")
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
        self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
        self.objpoints, self.imgpoints_r, img_shape, None, None)
        print("stereo calibrate")
        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
        cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        self.objpoints, self.imgpoints_l,
        self.imgpoints_r, self.M1, self.d1, self.M2,
        self.d2, dims,
        criteria=stereocalib_criteria, flags=flags)
        print("ret",ret)
        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        # WHAT IS R P: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
        # rectify stereo cameras, to make the plane parallel
        R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(M1, d1, M2, d2, dims, R, T)

        # camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
        # ('dist2', d2), ('rvecs1', self.r1),
        # ('rvecs2', self.r2), ('R', R), ('T', T),
        # ('E', E), ('F', F)])

        camera_model = {'M1': M1, 'M2': M2, 'dist1':d1,
        'dist2':d2, 'rvecs1':self.r1,
        'rvecs2':self.r2, 'R': R, 'T': T,
        'E': E, 'F': F, "R1":R1, "R2":R2,"P1":P1, "P2":P2,"Q":Q}
        # R1, R2, P1, P2 can be used to undistort image, see as follows:
        # https://stackoverflow.com/questions/27431062/stereocalibration-in-opencv-on-python
        # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
        """
        # https://www.programcreek.com/python/example/89319/cv2.initUndistortRectifyMap
        def compute_stereo_rectification_maps(stereo_rig, im_size, size_factor):
            new_size = (int(im_size[1] * size_factor), int(im_size[0] * size_factor))
            rotation1, rotation2, pose1, pose2 = \
                cv2.stereoRectify(cameraMatrix1=stereo_rig.cameras[0].intrinsics.intrinsic_mat,
                                distCoeffs1=stereo_rig.cameras[0].intrinsics.distortion_coeffs,
                                cameraMatrix2=stereo_rig.cameras[1].intrinsics.intrinsic_mat,
                                distCoeffs2=stereo_rig.cameras[1].intrinsics.distortion_coeffs,
                                imageSize=(im_size[1], im_size[0]),
                                R=stereo_rig.cameras[1].extrinsics.rotation,
                                T=stereo_rig.cameras[1].extrinsics.translation,
                                flags=cv2.CALIB_ZERO_DISPARITY,
                                newImageSize=new_size
                                )[0:4]
            map1x, map1y = cv2.initUndistortRectifyMap(stereo_rig.cameras[0].intrinsics.intrinsic_mat,
                                                    stereo_rig.cameras[0].intrinsics.distortion_coeffs,
                                                    rotation1, pose1, new_size, cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(stereo_rig.cameras[1].intrinsics.intrinsic_mat,
                                                    stereo_rig.cameras[1].intrinsics.distortion_coeffs,
                                                    rotation2, pose2, new_size, cv2.CV_32FC1)
            return map1x, map1y, map2x, map2y 
        """
        cv2.destroyAllWindows()
        return camera_model

cal = StereoCalibration('./video2')
print(cal.camera_model)
import pickle

with open('cam_model.pickle', 'wb') as handle:
    pickle.dump(cal.camera_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('cam_model.pickle', 'rb') as handle:
    cal.camera_model = pickle.load(handle)