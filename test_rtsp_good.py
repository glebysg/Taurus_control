
import threading
import cv2

class RTSCapture(cv2.VideoCapture):
    """Real Time Streaming Capture.
    这个类必须使用 RTSCapture.create 方法创建，请不要直接实例化
    """

    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"] #用于识别实时流

    @staticmethod
    def create(url, *schemes):
        """实例化&初始化
        rtscap = RTSCapture.create("rtsp://example.com/live/1")
        or
        rtscap = RTSCapture.create("http://example.com/live/1.m3u8", "http://")
        """
        rtscap = RTSCapture(url)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True)
        rtscap.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            # 这里可能是本机设备
            pass

        return rtscap

    def isStarted(self):
        """替代 VideoCapture.isOpened() """
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):
        """子线程读取最新视频帧方法"""
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        """读取最新视频帧
        返回结果格式与 VideoCapture.read() 一样
        """
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):
        """启动子线程读取视频帧"""
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        """退出子线程方法"""
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()



import sys

if __name__ == '__main__':
    recording=False

    #rtsp://192.168.1.164:13940/stream1"
    #rtsp://192.168.1.165:13950/stream1
    # rtscap = RTSCapture.create("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
    rtscap0 = RTSCapture.create("rtsp://192.168.1.164:13940/stream1")
    rtscap1 = RTSCapture.create("rtsp://192.168.1.165:13950/stream1")
    rtscap0.start_read() #启动子线程并改变 read_latest_frame 的指向
    rtscap1.start_read()
    cnt0, cnt1 = 0,0
    while rtscap0.isStarted():
        ok0, frame0 = rtscap0.read_latest_frame() #read_latest_frame() 替代 read()
        ok1, frame1 = rtscap1.read_latest_frame() #read_latest_frame() 替代 read()
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        if ok0:
            cv2.imshow("cam0", frame0)
            if recording:
                cv2.imwrite(f"cam0/{cnt0}.jpg",frame0)
                cnt0+=1
        if ok1:
            cv2.imshow("cam1", frame1)
            if recording:
                cv2.imwrite(f"cam1/{cnt1}.jpg",frame1)
                cnt1+=1

    rtscap0.stop_read()
    rtscap0.release()
    rtscap1.stop_read()
    rtscap1.release()
    cv2.destroyAllWindows()