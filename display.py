import sys
from threading import Thread 
import socket
import numpy as np
import sys
from datetime import datetime
import json 
import time
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pynput
from pynput.keyboard import Key, Listener
import threading
import copy
import pickle as pkl

global last_right
global last_left
global pose_id
last_left = None
last_right = None


def on_press(key):
    global last_right
    global last_left
    global pose_id
    selected_coords = [[0,0], [2,2], [1,5], [5,1], [4,3], [8,3],
                       [4,0], [1,2], [7,1], [3,4], [6,5], [7,3]]    

    pressed_key = None
    try:
        pressed_key = key.char     
    except AttributeError:
        pressed_key = key

    if pressed_key == 's':
        filename="data/homography/poses_%s" %(point_id)
        # save pkl object
        with open(filename, "wb") as output_file:
            pkl.dump(np.concatenate((last_left,last_right)),output_file)
    elif pressed_key == 'n':
        print("calibration for coordinate (%d,%d), id %d:" %(selected_coords[pose_id-1][0], 
            selected_coords[pose_id-1][1], pose_id))
        pose_id == min(12, pose_id+1)
    elif pressed_key == 'p':
        print("calibration for coordinate (%d,%d), id %d:" %(selected_coords[pose_id-1][0], 
            selected_coords[pose_id-1][1], pose_id))
        pose_id == max(1, pose_id-1)
    elif pressed_key == 'd':
        print("calibration for coordinate (%d,%d), id %d:" %(selected_coords[pose_id-1][0], 
            selected_coords[pose_id-1][1], pose_id))
        print("pose left", last_left)
        print("pose right", last_right)

def receive_poses():
    global last_right
    global last_left
    global pose_id
    pose_id = 1
    HOST = '128.46.103.192'  # Standard loopback interface address (localhost)
    PORT = 9753        # Port to listen on (non-privileged ports are > 1023)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))
        while True:
            print("get poses")
            data, server = s.recvfrom(4096)
            if not data:
                break
            pkg = json.loads(data)
            current_state=pkg.get("current_state", -1)
            left_data=pkg.get("arm0", "").split(" ")
            right_data=pkg.get("arm1", "").split(" ")
            parsed=0
            pkg = json.loads(data)
            current_state=pkg.get("current_state", -1)
            left_data=pkg.get("arm0", "").split(" ")
            right_data=pkg.get("arm1", "").split(" ")
            # if left_data != [''] and right_data != ['']:
            #     last_left = copy.copy(left_data)
            #     last_rigt = copy.copy(right_data)
            # print(len(left_data), len(right_data))
            # PARSE LEFT POSE
            if len(left_data)==44:
                # data in microns and radians
                left_pose_string = np.array(left_data[8:20])
                # flattened rotation matrix of the tip pose (flattened by row)
                # without the 0 0 0 1 useless row
                left_pose = left_pose_string.astype(np.float)
                last_left = np.copy(left_pose)
            # PARSE RIGHT POSE
            if len(right_data)==44:
                # data in microns and radians
                right_pose_string = np.array(right_data[8:20])
                # flattened rotation matrix of the tip pose (flattened by row)
                # without the 0 0 0 1 useless row
                right_pose =right_pose_string.astype(np.float)
                last_right = np.copy(right_pose)

if __name__ == "__main__":
    socket_thread = threading.Thread(target=receive_poses,args=())
    socket_thread.start()
    with Listener(on_press = on_press) as listener:
        listener.join()



