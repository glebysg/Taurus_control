"""
usage: 
1. generate dot files
2. visualize by http://webgraphviz.com/
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from scipy.spatial.transform import Rotation as R
from  collections import defaultdict

# f = "./logs/contact2_good.txt"
f = "./logs/bottle_haptic_on.txt"
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
df.sort_values(by=["time"])
# print(df["time"]-df["time"][0])

df["timediff"] = df["time"].diff()
df["error"] = (df['real_x'] - df['target_x'])**2 + (df['real_y'] - df['target_y'])**2 + (df['real_z'] - df['target_z'])**2
df["error"]  = df["error"]  ** 0.5
df["error_x"] = df['real_x'] - df['target_x']
df["error_y"] = df['real_y'] - df['target_y']
df["error_z"] = df['real_z'] - df['target_z']

df["vel_x"] = df['real_x'].diff()
df["vel_y"] = df['real_y'].diff()
df["vel_z"] = df['real_z'].diff()
df["vel"] = df['vel_x']**2 + df['vel_y']**2 + df['vel_z']**2 
df["vel"] = df["vel"] **0.5

df["vel_x"]/=df["timediff"]
df["vel_y"]/=df["timediff"]
df["vel_z"]/=df["timediff"]


corrs = []

for t, ex,ey,ez, vx,vy,vz in zip(df["time"], df["error_x"],df["error_y"],df["error_z"],df["vel_x"],df["vel_y"],df["vel_z"]):
    e = np.array([-ex,-ey,-ez])
    v = np.array([vx,vy,vz])
    corr = 1-(e*v).sum()/(1e-6+np.linalg.norm(e)*np.linalg.norm(v))
    corrs.append(corr)

        

df["corrs"]=corrs

algo="2"
corres_switch = []
if algo=="1":
    acc=0
    for e,v, corr in zip(df["error"], df["vel"], df["corrs"]):
        if corr*e>3100:
            cnt+=1
            if cnt>5:
                cnt=5
        else:
            cnt-=1
            if cnt<0:
                cnt=1
        if cnt>=5:
            corres_switch.append(1)
        else:
            corres_switch.append(0)
            
if algo=="2":
    accsum=0
    acc_sum=[]
    for e,v, corr in zip(df["error"], df["vel"], df["corrs"]):
        if np.isnan(corr):
            corres_switch.append(0)
            acc_sum.append(0)
            continue
        accsum += corr*e
        if accsum>8000:
            accsum=8000
        accsum -=1500
        acc_sum.append(accsum)
        if accsum<0:
            accsum=0
        if accsum>10000:
            accsum=10000
        if accsum>5000:
            corres_switch.append(1)
        else:
            corres_switch.append(0)
    df["accsum"]=acc_sum
    # save_img("img/accsum.png", df["time"],df["accsum"])

df["corres_switch"]=corres_switch
df = df.dropna()
y = df["corres_switch"]
# feature_name=['vel_x', 'vel_y','vel_z', "error_x","error_y","error_z"]
feature_name=['vel', "error", 'corrs']
feature_name=['vel', "error"]
# feature_name=['accsum', "error", 'corrs']
X = df[feature_name] 
print(X)
print(y)

from sklearn import tree

# dt = tree.DecisionTreeClassifier(criterion='entropy')
dt = tree.DecisionTreeClassifier(max_depth=3)

dt.fit(X, y)
tree.plot_tree(dt) 
pred = dt.predict(X)
print((pred>0.5).sum())
dotfile = open("dt.dot", 'w')
tree.export_graphviz(dt, out_file=dotfile, feature_names=feature_name)
dotfile.close()