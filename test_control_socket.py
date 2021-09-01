import socket
import sys
import json 
import numpy as np
import time

#packet = r'{"cmd":"1s","rot1" : [[0.701, 0, 0.707], [0, 1, 0], [0.707, 0, 0.707]], "pos1": [1,2,3], "gripper1":30}'
packet = r'{"cmd":"+1s","rot1" : [0.701, 0, 0.707, 0, 1, 0, 0.707, 0, 0.707], "pos1": [1,2,3], "gripper1":30}'

def packet_xyz(g1,x1,y1,z1):
    p = r'{"gripper0":%d,"pos0":[%d,%d,%d]}' % (g1,x1,y1,z1)
    return str.encode(p)

def packet_gripper(arg0,arg1):
    p = r'{"gripper0":%d,"gripper1":%d}' % (arg0,arg1)
    return str.encode(p)
center=(358079,-82799,111300)
server = ("192.168.1.118", 8642)
bufferSize          = 1024

 
# Create a UDP socket at client side
sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
i=0
k=0
sign =1
while True:
    cmd = [(i/20)%100]
    if sign>0:
        cmd.extend([p+2000 for p in center])
    else:
        cmd.extend([p-2000 for p in center])

    pkt = packet_xyz(*cmd)
    # pkt = packet_gripper(i,i)
    # pkt = str.encode(packet)
    i+=1 * sign
    if abs(i)>1000:
        sign = -sign
    if i==0:
        sign=-sign
    # print(pkt)
    sock.sendto(pkt, server)
    time.sleep(0.01);
    k+=1
    if i%100==0:
        print(i,k)
    