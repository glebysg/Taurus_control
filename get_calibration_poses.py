import logging
import time
import os
import unittest
import numpy as np
import copy
import sys
from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import IPython
import argparse
import pickle as pkl

'''
    THIS FILE SAVED THE POSES OF THE YUMI ARM
    USE YUMI ROBOT IN MODE LEAD-THROUGH TO MOVE THE ARM TO THE POINT YOU WISH TO MEASURE
    THEN (WHILE SERVER IS RUNNING) RUN THIS SCRIPT WITH PARAMETERS ARM AND POINT NUMBER
    THE SCRIPT WILL SAVE THE FILE "poses_%s_%s (arm, point_id)

'''

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action="store", dest="arm", default="left",
            help="record arm. Possible options: left, right or both")
    parser.add_argument('-i', action="store", dest="point_id", default="0", type=int,
            help="record point. possible options 1 to 12")
    # parser.add_argument('-s', action="store", dest="filename", default="pose",
    #         help="name of the pkl file where the object is recorded")

    selected_coords = [[0,0], [2,2], [1,5], [5,1], [4,3], [8,3],
                       [4,0], [1,2], [7,1], [3,4], [6,5], [7,3]]

    args = parser.parse_args()
    pt_id = args.point_id
    print "calibration for arm %s, coordinate (%d,%d), id %d:" %(args.arm,
                    selected_coords[pt_id-1][0], selected_coords[pt_id-1][1], pt_id)

    if pt_id<=6:
        print("Use the flat level")
    else:
        print("Use the raised level")

    # point_id = raw_input("Number of point you want to record:")  # Python 2
    # print text

    # Initialize Yumi
    y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)

    #setup the ool distance for the surgical grippers
    # ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])
    DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
    DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
    y.left.set_tool(DELTALEFT)
    y.right.set_tool(DELTARIGHT)
    y.set_v(40)
    y.set_z('z100')

    # Get Poses
    poses = {'left': None, 'right':None}
    if args.arm == "left" or args.arm == "both":
        poses['left']=y.left.get_pose()
    if args.arm == "right" or args.arm == "both":
    	poses['right']=y.right.get_pose()
    print poses

    filename="data/homography/poses_%s_%s" %(args.arm,args.point_id)

    # save pkl object
    with open(filename, "wb") as output_file:
        pkl.dump(poses,output_file)



if __name__ == '__main__':
    main()

