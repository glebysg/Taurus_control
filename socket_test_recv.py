import socket
import sys
import json 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
def update_line(hl, new_data):
    xdata, ydata, zdata = hl._verts3d
    hl.set_xdata(list(np.append(xdata, new_data[0])))
    hl.set_ydata(list(np.append(ydata, new_data[1])))
    hl.set_3d_properties(list(np.append(zdata, new_data[2])))
    plt.draw()

map = plt.figure()
map_ax = Axes3D(map)
map_ax.autoscale(enable=True, axis='both', tight=True)
 
# # # Setting the axes properties
map_ax.set_xlim3d([-0.5, 0.5])
map_ax.set_ylim3d([-0.5, 0.5])
map_ax.set_zlim3d([-0.5, 0.5])
hl, = map_ax.plot3D([0], [0], [0])
 
# update_line(hl, (2,2, 1))
# plt.show(block=False)
# plt.pause(1)
 
# update_line(hl, (5,5, 5))
# plt.show(block=False)
# plt.pause(2)
 
# update_line(hl, (8,1, 4))
# plt.show(block=True)

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

address = ('192.168.1.118', 9753)
message = b'This is the message.  It will be repeated.'
sock.bind(address)
print('waiting to receive')

def parse_pkg(pkg):
    
    MEGA=1e6
    pkg_time = pkg["timestamp"]
    left_data = pkg["arm0"].split(" ")
    right_data = pkg["arm1"].split(" ")
    
    ret={"pkg_time":pkg_time}
    for idx, data in enumerate([left_data, right_data]):
        ret[(idx, "time")] = data[0]
        ret[(idx,"R")] = np.array([float(i)/MEGA for i in data[1:10]]).reshape((3,3))
        ret[(idx,"t")] = np.array([float(i)/MEGA for i in data[10:13]])
        ret[(idx,"xyz")] = np.array([float(i)/MEGA for i in data[13:16]])
        ret[(idx,"rpy")] = np.array([float(i)/MEGA for i in data[16:19]])
        ret[(idx,"gripper")]  = data[19]
    return ret
try:

    # # Send data
    # print('sending {!r}'.format(message))
    # sent = sock.sendto(message, server_address)

    # Receive response
    i=0
    while True:
        data, server = sock.recvfrom(4096)
        print(data)
        continue
        pkg = json.loads(data)
        
        robotdata = parse_pkg(pkg)
        i+=1
        if i%50==0:
            plt.pause(0.001)
            plt.show(block=False)
            print(i)
        # print(robotdata[(0,"xyz")], robotdata[(1,"xyz")])
        # print(robotdata)
        update_line(hl, robotdata[(0,"xyz")])


finally:
    print('closing socket')
    sock.close()