import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget,QLineEdit,QPushButton 
from PyQt5.QtCore import QSize    
from PyQt5.QtCore import pyqtSlot, QRunnable, QThread, QThreadPool, pyqtSignal
from threading import Thread 
import socket
import numpy as np
import sys
from datetime import datetime
import json 
import time
from scipy.spatial.transform import Rotation as R
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
class RecvThread(QThread):
    update_signal = QtCore.pyqtSignal(bytes)
    def __init__(self): 
        super().__init__() 
        # self.window=window
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        address = ('128.46.103.192', 9753)
        self.sock.bind(address)

    def run(self): 
        cnt=0
        t = time.time()
        print('Started running .... ')
        while True:
            t2 = time.time()
            data, server = self.sock.recvfrom(4096) # ,server
            # print("Data is here ", data)
            if t2-t > 0.02:
                self.update_signal.emit(data)
                t = t2
            
            

           

class HelloWindow(QMainWindow):
    def __init__(self):
        self.current_state = -1
        self.save_handle = None
        self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(640, 480))    
        self.setWindowTitle("Taurus") 

        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   

        gridLayout = QGridLayout(self)     
        centralWidget.setLayout(gridLayout)  
        self.xe1,self.ye1,self.ze1 = QLineEdit("358070"), QLineEdit("-82800"), QLineEdit("111300")
        self.xe2,self.ye2,self.ze2 = QLineEdit("358080"), QLineEdit("82800"), QLineEdit("111300")
        self.roll1,self.pitch1,self.yaw1 = QLineEdit("100.81"), QLineEdit("-65.53"), QLineEdit("77.53")
        self.roll2,self.pitch2,self.yaw2 = QLineEdit("-100.81"), QLineEdit("-65.53"), QLineEdit("-77.53")
        self.gripper1, self.gripper2 = QLineEdit("0"), QLineEdit("0")
        self.movement_time = QLineEdit("0.5")
        gridLayout.addWidget(QLabel("X"), 0, 0)
        gridLayout.addWidget(self.xe1, 0, 1)
        gridLayout.addWidget(self.xe2, 0, 2)

        gridLayout.addWidget(QLabel("Y"), 1, 0)
        gridLayout.addWidget(self.ye1, 1, 1)
        gridLayout.addWidget(self.ye2, 1, 2)
        
        gridLayout.addWidget(QLabel("Z"), 2, 0)
        gridLayout.addWidget(self.ze1, 2, 1)
        gridLayout.addWidget(self.ze2, 2, 2)

        gridLayout.addWidget(QLabel("Roll"), 3, 0)
        gridLayout.addWidget(self.roll1, 3, 1)
        gridLayout.addWidget(self.roll2, 3, 2)

        gridLayout.addWidget(QLabel("Pitch"), 4, 0)
        gridLayout.addWidget(self.pitch1, 4, 1)
        gridLayout.addWidget(self.pitch2, 4, 2)

        gridLayout.addWidget(QLabel("Yaw"), 5, 0)
        gridLayout.addWidget(self.yaw1, 5, 1)
        gridLayout.addWidget(self.yaw2, 5, 2)

        gridLayout.addWidget(QLabel("Gr"), 6, 0)
        gridLayout.addWidget(self.gripper1, 6, 1)
        gridLayout.addWidget(self.gripper2, 6, 2)

        send_button = QPushButton('Send', self)
        stow_button = QPushButton('Stow', self)
        approach_button = QPushButton('Approach', self)
        safe_button = QPushButton('Safe', self)
        display_button = QPushButton('Display', self)
        self.savelog_button =savelog_button= QPushButton('SaveLog', self)

        send_button.clicked.connect(self.sendbtn)
        approach_button.clicked.connect(self.approachbtn)
        safe_button.clicked.connect(self.safebtn)
        stow_button.clicked.connect(self.stowbtn)
        display_button.clicked.connect(self.displaybtn)
        savelog_button.clicked.connect(self.savebtn)

        gridLayout.addWidget(send_button, 7, 0)
        gridLayout.addWidget(stow_button, 7, 1)
        gridLayout.addWidget(approach_button, 7, 2)
        gridLayout.addWidget(safe_button, 7, 3)
        gridLayout.addWidget(display_button, 7, 4)
        gridLayout.addWidget(savelog_button, 7, 5)
        gridLayout.addWidget(QLabel("current_state"), 8, 0)
        self.current_state = QLabel("")
        gridLayout.addWidget(self.current_state, 8, 1)
        self.x1_disp,self.y1_disp,self.z1_disp = QLabel(""),QLabel(""),QLabel("")
        self.x2_disp,self.y2_disp,self.z2_disp = QLabel(""),QLabel(""),QLabel("")
        self.roll1_disp,self.pitch1_disp,self.yaw1_disp = QLabel(""),QLabel(""),QLabel("")
        self.roll2_disp,self.pitch2_disp,self.yaw2_disp = QLabel(""),QLabel(""),QLabel("")
        self.xroll1_disp,self.xpitch1_disp,self.xyaw1_disp = QLabel(""),QLabel(""),QLabel("")
        self.xroll2_disp,self.xpitch2_disp,self.xyaw2_disp = QLabel(""),QLabel(""),QLabel("")
        self.xx1_disp,self.yy1_disp,self.zz1_disp = QLabel(""),QLabel(""),QLabel("")
        self.xx2_disp,self.yy2_disp,self.zz2_disp = QLabel(""),QLabel(""),QLabel("")
        self.Gr1_disp, self.Gr2_disp = QLabel(""),QLabel("")
        self.accsum_left_disp, self.accsum_right_disp = QLabel(""),QLabel("")
        self.contact_switch_left_disp, self.contact_switch_right_disp = QLabel(""),QLabel("")
        gridLayout.addWidget(QLabel("Xreal"), 9, 0)
        gridLayout.addWidget(self.x1_disp, 9, 1)
        gridLayout.addWidget(self.x2_disp, 9, 2)


        gridLayout.addWidget(QLabel("Yreal"), 10, 0)
        gridLayout.addWidget(self.y1_disp, 10, 1)
        gridLayout.addWidget(self.y2_disp, 10, 2)


        gridLayout.addWidget(QLabel("Zreal"), 11, 0)
        gridLayout.addWidget(self.z1_disp, 11, 1)
        gridLayout.addWidget(self.z2_disp, 11, 2)

        gridLayout.addWidget(QLabel("Roll"), 12, 0)
        gridLayout.addWidget(self.roll1_disp, 12, 1)
        gridLayout.addWidget(self.roll2_disp, 12, 2)

        gridLayout.addWidget(QLabel("Pitch"), 13, 0)
        gridLayout.addWidget(self.pitch1_disp, 13, 1)
        gridLayout.addWidget(self.pitch2_disp, 13, 2)

        gridLayout.addWidget(QLabel("Yaw"), 14, 0)
        gridLayout.addWidget(self.yaw1_disp, 14, 1)
        gridLayout.addWidget(self.yaw2_disp, 14, 2)

        gridLayout.addWidget(QLabel("Gr"), 15, 0)
        gridLayout.addWidget(self.Gr1_disp, 15, 1)
        gridLayout.addWidget(self.Gr2_disp, 15, 2)

        gridLayout.addWidget(QLabel("xx"), 16, 0)
        gridLayout.addWidget(self.xx1_disp, 16, 1)
        gridLayout.addWidget(self.xx2_disp, 16, 2)

        gridLayout.addWidget(QLabel("yy"), 17, 0)
        gridLayout.addWidget(self.yy1_disp, 17, 1)
        gridLayout.addWidget(self.yy2_disp, 17, 2)

        gridLayout.addWidget(QLabel("zz"), 18, 0)
        gridLayout.addWidget(self.zz1_disp, 18, 1)
        gridLayout.addWidget(self.zz2_disp, 18, 2)

        gridLayout.addWidget(QLabel("xroll"), 19, 0)
        gridLayout.addWidget(self.xroll1_disp, 19, 1)
        gridLayout.addWidget(self.xroll2_disp, 19, 2)

        gridLayout.addWidget(QLabel("xpitch"), 20, 0)
        gridLayout.addWidget(self.xpitch1_disp, 20, 1)
        gridLayout.addWidget(self.xpitch2_disp, 20, 2)

        gridLayout.addWidget(QLabel("xyaw"), 21, 0)
        gridLayout.addWidget(self.xyaw1_disp, 21, 1)
        gridLayout.addWidget(self.xyaw2_disp, 21, 2)

        gridLayout.addWidget(QLabel("Accsum"), 22, 0)
        gridLayout.addWidget(self.accsum_left_disp, 22, 1)
        gridLayout.addWidget(self.accsum_right_disp, 22, 2)

        gridLayout.addWidget(QLabel("contact_sw"), 23, 0)
        gridLayout.addWidget(self.contact_switch_left_disp, 23, 1)
        gridLayout.addWidget(self.contact_switch_right_disp, 23, 2)

        gridLayout.addWidget(QLabel("movement_time"), 24, 0)
        gridLayout.addWidget(self.movement_time, 24, 1)

        self.accsum_left=0
        self.accsum_right=0
        self.real_position_left_last=None
        self.real_position_right_last=None
        

        self.recvthread = RecvThread()
        self.recvthread.update_signal.connect(self.package_callback)
        self.recvthread.start()

    def to_rpy(self, rmat_list):
        rmat_list = [i/1e6 for i in rmat_list]
        rot = np.array(rmat_list).reshape(3,3)
        rot = R.from_dcm(rot)
        y,p,r = rot.as_euler('ZYX', degrees=True)
        return r,p,y

    def package_callback(self, data):
        pkg = json.loads(data)

        self.current_state.setText(str(pkg.get("current_state", -1)) )
        left_data=pkg.get("arm0", "").split(" ")
        right_data=pkg.get("arm1", "").split(" ")
        parsed=0
        
        if len(left_data)==33 :
            #ValueError: could not convert string to float: '1.#QNAN'
            try:
                self.left_data= left_data=[float(x) for x in left_data]
            except:
                return 
            roll1,pitch1,yaw1 = self.to_rpy(left_data[1:10])
            x1,y1,z1 = left_data[10:13]
            xx,yy,zz = left_data[13:16]
            xpitch, xyaw, xroll = np.array(left_data[16:19])/(1e6*np.pi/180) 
            gripper = left_data[19]
            target_position = left_data[20:23]
            target_pose = self.to_rpy(left_data[23:32])
            target_opendeg = left_data[32]
            ################################################
            #### left and right arm seems reversed for contact detect..
            real_position_left = np.array([x1,y1,z1])

            if self.real_position_left_last is None:
                self.real_position_left_last = real_position_left
            e = e_left = target_position - real_position_left
            v = v_left = real_position_left - self.real_position_left_last
            corr = 1-(e*v).sum()/(1e-6+np.linalg.norm(e)*np.linalg.norm(v))
            # print(corr, corr*np.linalg.norm(e))
            self.real_position_left_last = real_position_left
            e2 = np.linalg.norm(e)-1500
            if e2<0: e2=0
            self.accsum_left += corr*np.linalg.norm(e)
            if self.accsum_left>4000:
                self.accsum_left=4000
            self.accsum_left -=500
            if self.accsum_left<0:
                self.accsum_left=0
            if self.accsum_left>3000:
                contact_switch_left = 1
            else:
                contact_switch_left = 0
            ###########################################
            self.x1_disp.setText(str(target_position[0]))
            self.y1_disp.setText(str(target_position[1]))
            self.z1_disp.setText(str(target_position[2]))
            self.roll1_disp.setText(str(target_pose[0]))
            self.pitch1_disp.setText(str(target_pose[1]))
            self.yaw1_disp.setText(str(target_pose[2]))
            self.Gr1_disp.setText(str(gripper))
            self.xx1_disp.setText(str(xx))
            self.yy1_disp.setText(str(yy))
            self.zz1_disp.setText(str(zz))
            self.xroll1_disp.setText(str(xroll))
            self.xpitch1_disp.setText(str(xpitch))
            self.xyaw1_disp.setText(str(xyaw))
            self.accsum_left_disp.setText(str(self.accsum_left))
            self.contact_switch_left_disp.setText(str(contact_switch_left))
            parsed+=1
                # print(left_data[16:19])

        if len(right_data)==33:
            try:
                self.right_data=right_data=[float(x) for x in right_data]
            except:
                return
            roll2,pitch2,yaw2 = self.to_rpy(right_data[1:10])
            x2,y2,z2 = right_data[10:13]
            xx,yy,zz = right_data[13:16]
            xpitch, xyaw, xroll = np.array(right_data[16:19])/(1e6*np.pi/180) 
            gripper = right_data[19]
            target_position = right_data[20:23]
            target_pose = self.to_rpy(right_data[23:32])
            target_opendeg = right_data[32]

            ################################################
            #### left and right arm seems reversed for contact detect..
            real_position_right = np.array([x2,y2,z2])

            if self.real_position_right_last is None:
                self.real_position_right_last = real_position_right
            e = e_right = target_position - real_position_right
            v = v_right = real_position_right - self.real_position_right_last
            corr = 1-(e*v).sum()/(1e-6+np.linalg.norm(e)*np.linalg.norm(v))
            # print(corr, corr*np.linalg.norm(e))
            self.real_position_right_last = real_position_right
            e2 = np.linalg.norm(e)-1500
            if e2<0: e2=0
            self.accsum_right += corr*e2
            if self.accsum_right>4000:
                self.accsum_right=4000
            self.accsum_right -=500
            if self.accsum_right<0:
                self.accsum_right=0
            if self.accsum_right>3000:
                contact_switch_right = 1
            else:
                contact_switch_right = 0
            ###########################################

            self.x2_disp.setText(str(target_position[0]))
            self.y2_disp.setText(str(target_position[1]))
            self.z2_disp.setText(str(target_position[2]))
            self.roll2_disp.setText(str(target_pose[0]))
            self.pitch2_disp.setText(str(target_pose[1]))
            self.yaw2_disp.setText(str(target_pose[2]))

            self.Gr2_disp.setText(str(gripper))

            self.xx2_disp.setText(str(xx))
            self.yy2_disp.setText(str(yy))
            self.zz2_disp.setText(str(zz))
            self.xroll2_disp.setText(str(xroll))
            self.xpitch2_disp.setText(str(xpitch))
            self.xyaw2_disp.setText(str(xyaw))
            self.accsum_right_disp.setText(str(self.accsum_right))
            self.contact_switch_right_disp.setText(str(contact_switch_right))

            parsed+=1
        # print(parsed)
        if parsed==2:
            if self.save_handle is not None:
                self.save_handle.write(data.decode("utf-8")+"\n")

    def sendbtn(self):
        print('PyQt5 button click')
        pkg = {}
        pkg["pos0"] = [int(self.xe1.text()), int(self.ye1.text()),int(self.ze1.text())]
        pkg["pos1"] = [int(self.xe2.text()), int(self.ye2.text()),int(self.ze2.text())]
        pkg["gripper0"] = int(self.gripper1.text())
        pkg["gripper1"] = int(self.gripper2.text())
        rot1 = np.array([float(self.yaw1.text()), float(self.pitch1.text()), float(self.roll1.text())])
        rot2 = np.array([float(self.yaw2.text()), float(self.pitch2.text()), float(self.roll2.text())])
        rot1 = R.from_euler("ZYX", rot1, degrees=True).as_dcm().flatten()
        rot2 = R.from_euler("ZYX", rot2, degrees=True).as_dcm().flatten()
        pkg["rot0"] = list(rot1)
        pkg["rot1"] = list(rot2)
        pkg["movement_time"] = float(self.movement_time.text())
        
        pkt=json.dumps(pkg)
        pkt=str.encode(pkt)
        print(pkt)
        self.sock.sendto(pkt, ("128.46.103.80", 8642))

    def approachbtn(self):
        print("sending approach command")
        pkg= {}
        pkg["cmd"] = "Approach"
        pkt=json.dumps(pkg)
        pkt=str.encode(pkt)
        print(pkt)
        self.sock.sendto(pkt, ("128.46.103.80", 8642))
     
    
    def safebtn(self):
        print("sending safe command")
        pkg= {}
        pkg["cmd"] = "Safe"
        pkt=json.dumps(pkg)
        pkt=str.encode(pkt)
        print(pkt)
        self.sock.sendto(pkt, ("128.46.103.80", 8642))

    def savebtn(self):
        if self.save_handle is None:
            filename = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.save_handle = open(filename+".txt", "w")
            self.savelog_button.setText("DisableSaveLog")
        else:
            self.save_handle.close()
            self.save_handle = None
            self.savelog_button.setText("SaveLog")
    
    def stowbtn(self):
        print("sending stow command")
        pkg= {}
        pkg["cmd"] = "Stowed"
        pkt=json.dumps(pkg)
        pkt=str.encode(pkt)
        print(pkt)
        self.sock.sendto(pkt, ("128.46.103.80", 8642))

    def display_emitter(self):
        last = None
        while True:
            x1,y1,z1 = self.left_data[10:13]
            x2,y2,z2 = self.right_data[10:13]
            if last is None:
                yield 0
                last=x2
            else:
                yield x2-last
                last=x2
    def displaybtn(self):
        from osc_display import Scope
        np.random.seed(19680801)


        fig, ax = plt.subplots()
        scope = Scope(ax)

        # pass a generator in "emitter" to produce data for the update func
        ani = animation.FuncAnimation(fig, scope.update, self.display_emitter, interval=10,
                              blit=True)

        plt.show()
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = HelloWindow()
    mainWin.show()
    sys.exit( app.exec_() )