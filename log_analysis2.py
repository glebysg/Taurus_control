# libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from scipy.spatial.transform import Rotation as R
from  collections import defaultdict
# f = "./logs/left_hand_bottle.txt"
# f = "./logs/both_hand_book.txt"
# f = "./logs/no_obj.txt"
f = "./logs/bottle_haptic2.txt"
arm = "arm0"
datadict=defaultdict(list)
def to_rpy(rmat_list):
    rmat_list = [i/1e6 for i in rmat_list]
    rot = np.array(rmat_list).reshape(3,3)
    rot = R.from_dcm(rot)
    y,p,r = rot.as_euler('ZYX', degrees=True)
    return r,p,y

def add_data(datadict, new_dict):
    for k in new_dict:
        datadict[k].append(new_dict[k])
        
def parse(arm_array):
    arm_array = [float(i) for i in arm_array.split(" ")]
    timestep = arm_array[0]
    real_roll,real_pitch,real_yaw = to_rpy(arm_array[1:10])
    # x1,y1,z1 = arm_array[10:13]
    real_x,real_y,real_z = arm_array[13:16]
    # real_pitch, real_yaw, real_roll = np.array(arm_array[16:19])/(1e6*np.pi/180) 
    real_opendeg = arm_array[19]
    target_x,target_y,target_z = arm_array[20:23]
    target_roll, target_pitch, target_yaw = to_rpy(arm_array[23:32])
    target_opendeg = arm_array[32]
    return {"time":timestep,"real_x":real_x,"real_y":real_y, "real_z":real_z,
            "target_x":target_x,"target_y":target_y, "target_z":target_z,
            "real_pitch":real_pitch, "real_yaw":real_yaw, "real_roll":real_roll,
            "target_pitch":target_pitch, "target_yaw":target_yaw, "target_roll":target_roll,
            "real_opendeg":real_opendeg,
            "target_opendeg":target_opendeg}
    
with open(f,"r") as fd:
    while True:
        js = fd.readline()
        if js =="":
            break
        j = json.loads(js)
        add_data(datadict, parse(j[arm]))

df=pd.DataFrame(datadict)
# remove bad data
df = df[df["target_opendeg"]>1]
# A equaltion: speed not fast, error persist long (accumulate),error is not caused by target move
df["timediff"] = df["time"].diff()
df["target_move"] = df['target_x'].diff()**2 + df['target_y'].diff() ** 2 + df['target_z'].diff() ** 2
df["target_move"] = df["target_move"] ** 0.5
df["real_move"] = df['real_x'].diff()**2 + df['real_y'].diff() ** 2 + df['real_z'].diff() ** 2
df["real_move"] = df["real_move"] ** 0.5

df["error"] = (df['real_x'] - df['target_x'])**2 + (df['real_y'] - df['target_y'])**2 + (df['real_z'] - df['target_z'])**2
df["error"]  = df["error"]  ** 0.5
df["error_cumsum"] = df["error"].cumsum()
df["target_spd"] = df["target_move"] / df["timediff"]
df["real_spd"] = df["real_move"] / df["timediff"]
df["spd_ratio"] = df["target_spd"]/ df["real_spd"]
df["diff_real_roll"] = df["real_roll"].diff()
df["diff_real_pitch"] = df["real_pitch"].diff()
df["diff_real_yaw"] = df["real_yaw"].diff()
df["diff_real_roll"]  = df["diff_real_roll"] /df["timediff"]
df["diff_real_pitch"]  = df["diff_real_pitch"] /df["timediff"]
df["diff_real_yaw"]  = df["diff_real_yaw"] /df["timediff"]

df["diff_target_roll"] = df["target_roll"].diff()
df["diff_target_pitch"] = df["target_pitch"].diff()
df["diff_target_yaw"] = df["target_yaw"].diff()
df["diff_target_roll"]  = df["diff_target_roll"] /df["timediff"]
df["diff_target_pitch"]  = df["diff_target_pitch"] /df["timediff"]
df["diff_target_yaw"]  = df["diff_target_yaw"] /df["timediff"]

print(df)
# plot
fig = plt.figure()

plt.plot(df["time"],df["real_roll"]-df["target_roll"])
plt.plot(df["time"],df["real_pitch"]-df["target_pitch"])
plt.plot(df["time"],df["real_yaw"]-df["target_yaw"])

# plt.plot(df["time"],df["real_roll"])
# plt.plot(df["time"],df["real_pitch"])
# plt.plot(df["time"],df["real_yaw"])
# plt.plot(df["time"],df["target_roll"])
# plt.plot(df["time"],df["target_pitch"])
# plt.plot(df["time"],df["target_yaw"])

# plt.plot(df["time"],df["diff_real_roll"], label="diff real roll")
# plt.plot(df["time"],df["diff_real_pitch"], label="diff real pitch")
# plt.plot(df["time"],df["diff_real_yaw"], label="diff real yaw")

# plt.plot(df["time"],df["diff_target_roll"], label="diff target roll")
# plt.plot(df["time"],df["diff_target_pitch"], label="diff target pitch")
# plt.plot(df["time"],df["diff_target_yaw"], label="diff target yaw")

# plt.plot(df["time"],df["real_opendeg"]-df["target_opendeg"])

# plt.plot(df["time"],df["real_x"])
# plt.plot(df["time"],df["real_y"])
# plt.plot(df["time"],df["real_z"])
# plt.plot(df["time"],df["target_x"])
# plt.plot(df["time"],df["target_y"])
# plt.plot(df["time"],df["target_z"])
plt.legend()
plt.show()
