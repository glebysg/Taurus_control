import socket
import sys
import json 
import numpy as np
import time


center=(358079,-82799,111300)
server = ("192.168.1.118", 8642)
bufferSize  = 1024

def packet_xyz(g1,x1,y1,z1):
    p = r'{"gripper0":%d,"pos0":[%d,%d,%d]}' % (g1,x1,y1,z1)
    return str.encode(p)
 
# Create a UDP socket at client side
sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
i=0
k=0
sign =1
while True:
    cmd = [40]

    cmd.extend([p+30000 for p in center])

    pkt = packet_xyz(*cmd)
    sock.sendto(pkt, server)
    time.sleep(0.01);

    