import cv2
import time
# how to set 
cap = cv2.VideoCapture("rtsp://192.168.1.165:13950/stream1")
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()