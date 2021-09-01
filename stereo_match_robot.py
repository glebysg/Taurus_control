#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 
import pickle

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    mask = ~np.isnan(verts).any(axis=1)
    verts = verts[mask]
    colors = colors[mask]
    mask = ~np.isinf(verts).any(axis=1)
    verts = verts[mask]
    colors = colors[mask]
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def get_disparity(imgL, imgR):
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp


def read_images(video_left, video_right):
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
    return left_buf,right_buf
        
# https://blog.csdn.net/bcj296050240/article/details/52778741
# important:https://blog.csdn.net/bcj296050240/article/details/52778741
if __name__ == '__main__':
    print('loading images...')
    img_left, img_right = read_images(video_left="/home/ye/programming/TaurusCpp/python/video2/video_l.mp4",
                                      video_right="/home/ye/programming/TaurusCpp/python/video2/video_r.mp4")
    with open('cam_model.pickle', 'rb') as handle:
        cam = pickle.load(handle)
        M1, M2 = cam["M1"], cam["M2"]
        d1, d2 = cam["dist1"], cam["dist2"]
        r1, r2 = cam["rvecs1"], cam["rvecs2"]
        R, T = cam["R"], cam["T"]
        E, F = cam["E"], cam["F"]
        R1, R2 = cam["R1"], cam["R2"]
        P1, P2 = cam["P1"], cam["P2"]
        Q = cam["Q"]
        print(Q)
        
    # line 90:
    #https://github.com/MegaYEye/Taurus-Misc/blob/master/IE590%20Group%20Project/TaurusStereo/TaurusStereo/main.cpp
    # https://blog.csdn.net/bcj296050240/article/details/52778741
    
    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    # what is the homography rectification 
    # http://www.bmva.org/bmvc/2010/conference/paper89/abstract89.pdf
    # stereoRectifyUncalibrated
    # 3rd parameter: Optional rectification transformation in the object space (3x3 matrix). R1 or R2 , computed by stereoRectify can be passed here. 
    # 4th parameter:  In case of a stereo camera, newCameraMatrix is normally set to P1 or P2 computed by stereoRectify .
    map1x, map1y = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (1280, 720), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (1280, 720), cv2.CV_32FC1)

    imgU1 = cv2.remap(img_left[110], map1x, map1y, cv2.INTER_LINEAR)
    imgU2 = cv2.remap(img_right[110], map2x, map2y, cv2.INTER_LINEAR)
    
    # for i, (L, R) in enumerate(zip(img_left, img_right)):
    #     L = cv2.remap(L, map1x, map1y, cv2.INTER_LINEAR)
    #     R = cv2.remap(R, map2x, map2y, cv2.INTER_LINEAR)
    #     cv2.imwrite(f"stereo_rectified_L/{i}.jpg", L)
    #     cv2.imwrite(f"stereo_rectified_R/{i}.jpg", R)
    # pass
    # imgU1 = cv2.GaussianBlur(imgU1, (5, 5), 3)
    # imgU2 = cv2.GaussianBlur(imgU2, (5, 5), 3)
    # imgU1 = cv2.GaussianBlur(imgU1, (5, 5), 3)
    # imgU2 = cv2.GaussianBlur(imgU2, (5, 5), 3)
    # imgU1 = cv2.GaussianBlur(imgU1, (5, 5), 3)
    # imgU2 = cv2.GaussianBlur(imgU2, (5, 5), 3)
    # looks correct before here.
    
    # while True:
    #     img = np.concatenate([imgU1, imgU2], axis=1)
    #     cv2.imshow("CC",img)
    #     # cv2.imshow("L",imgU1)
    #     # cv2.imshow("R", imgU2)        
    #     # cv2.imshow("Lori",img_left[0])
    #     # cv2.imshow("Rori", img_right[0])
    #     cv2.waitKey(10)
    #     # cv2.imwrite("L.jpg",imgU1)
    #     # cv2.imwrite("R.jpg",imgU2)
    #     # exit()

    """
    	sgbm.preFilterCap = 63;
		sgbm.SADWindowSize = 9;//SADWindowSize > 0 ? SADWindowSize : 3;
		int cn = 1;
		sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
		sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
		sgbm.minDisparity = 0;
		sgbm.numberOfDisparities = numberOfDisparities;
		sgbm.uniquenessRatio = 1;
		sgbm.speckleWindowSize = 100;
		sgbm.speckleRange = 32;
		sgbm.disp12MaxDiff = 2;
		sgbm.fullDP = false;
    """

    # window_size = 3
    # min_disp = 32
    # num_disp = 112-min_disp
    # stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    #     preFilterCap = 63,
    #     numDisparities = num_disp,
    #     blockSize = 16,
    #     P1 = 8*3*window_size**2,
    #     P2 = 32*3*window_size**2,
    #     disp12MaxDiff = 1,
    #     uniquenessRatio = 10,
    #     speckleWindowSize = 100,
    #     speckleRange = 32
    # )
    # # why dividie by 16
    # print('computing disparity...')
    
    # stereo = cv2.StereoBM_create(
    #     # blockSize=5,
    #     # numDisparities=64,
    # )
    # imgU1=cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY)
    # imgU2=cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY)
    window_size = 9
	# left matcher is StereoSGBM_create
    stereo = cv2.StereoSGBM_create(
        numDisparities=96,
        blockSize=7,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=16,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = stereo.compute(imgU1, imgU2).astype(np.float32)/16 

    print('generating 3d point cloud...',)
    h, w = imgU1.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    # https://stackoverflow.com/questions/41503561/whats-the-difference-between-reprojectimageto3dopencv-and-disparity-to-3d-coo
    # however, directly define depth = fb/D needs to ensure the two cameras to be parallel to each other.
    # stereoRectify:The function computes the rotation matrices for each camera that (virtually) make both camera image planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate as input. As output, it provides two rotation matrices and also two projection matrices in the new coordinates. The function distinguishes the following two cases:
    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
    # can output Q	Output 4×4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgU1, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

    cv2.imshow('left', imgU1)
    cv2.imshow('disparity', (disp-disp.min()))
    cv2.waitKey()
    cv2.destroyAllWindows()
