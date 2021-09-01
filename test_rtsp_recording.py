
import threading
import cv2

class RTSCapture(cv2.VideoCapture):
    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"] 

    @staticmethod
    def create(url, *schemes):
        """
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
            
            pass

        return rtscap

    def isStarted(self):

        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):

        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):

        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):

        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
   
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()



import sys

if __name__ == '__main__':
    # recording tutorial
    # https://www.youtube.com/watch?v=1eHQIu4r0Bc
    #rtsp://192.168.1.164:13940/stream1"
    #rtsp://192.168.1.165:13950/stream1
    # rtscap = RTSCapture.create("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
    fps=5
    cnt_left=0
    cnt_right=0
    rtscap0 = RTSCapture.create("rtsp://192.168.1.164:13940/stream1")
    rtscap1 = RTSCapture.create("rtsp://192.168.1.165:13950/stream1")
    rtscap0.start_read() 
    rtscap1.start_read()
    init=True
    while rtscap0.isStarted() and rtscap1.isStarted():
        ok0, frame0 = rtscap0.read_latest_frame() #0 is right
        ok1, frame1 = rtscap1.read_latest_frame() #1 is left
        if ok0 and ok1 and init:
            init=False
            dim = (frame1.shape[1],frame1.shape[0])
            writer_left = cv2.VideoWriter("video_left.mp4", cv2.VideoWriter_fourcc(*'mp4v'),fps,dim)
            writer_right = cv2.VideoWriter("video_right.mp4", cv2.VideoWriter_fourcc(*'mp4v'),fps,dim)
            # writer_left = cv2.VideoWriter("video_left.avi", cv2.VideoWriter_fourcc(*'avc1'),fps,dim)
            # writer_right = cv2.VideoWriter("video_right.avi", cv2.VideoWriter_fourcc(*'avc1'),fps,dim)
        if init == False:
            if cv2.waitKey(1000//fps) & 0xFF == ord('q'):
                break
            if ok0 and ok1:
                cv2.imshow("left", frame1)
                cv2.imshow("right", frame0)
                writer_left.write(frame1)
                writer_right.write(frame0)
                cnt_left+=1
                cnt_right+=1
                
                
            # cnt+=1
            print(cnt_left, cnt_right)
            


    rtscap0.stop_read()
    rtscap0.release()
    rtscap1.stop_read()
    rtscap1.release()
    writer_left.release()
    writer_right.release()
    cv2.destroyAllWindows()