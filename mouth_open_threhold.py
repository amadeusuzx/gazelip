import cv2
import sys
import dlib
from PIL import Image
from imutils import face_utils
import numpy as np
import time
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

camera_id = 0
delay = 1
window_name = 'frame'
exp = -6
brightness = 10
cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_EXPOSURE, exp)
cap.set(cv2.CAP_PROP_BRIGHTNESS,brightness)
cap.set(cv2.CAP_PROP_FPS, 60)
def flipv(imgg):
    img2= np.zeros([480, 640, 3], np.uint8)
    for i in range(480):

        img2[i,:]=imgg[480-i-1,:]

    return img2
if not cap.isOpened():
    sys.exit()

tm = cv2.TickMeter()
tm.start()

count = 0
max_count = 10
fps = 0
angle=0
now_angle= 0
while cap.isOpened():
    ret, frame = cap.read()
    image = cv2.cvtColor(cv2.resize(
                frame, (120, 90)), cv2.COLOR_BGR2GRAY)
    rects = DETECTOR(image, 1)
    t1 = time.time()
    for (_, rect) in enumerate(rects):
        shape = PREDICTOR(image, rect)
        shape = face_utils.shape_to_np(shape)
    if rects:
        lip = cv2.boundingRect(shape[48:68])
        angle = np.linalg.norm(
            shape[62] - shape[66]) / np.linalg.norm(shape[60] - shape[64])
    if count == max_count:
        tm.stop()
        fps = max_count / tm.getTimeSec()
        tm.reset()
        tm.start()
        count = 0
        now_angle = angle

    cv2.putText(frame, 'FPS: {:.2f}'.format(fps),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv2.putText(frame, 'Angle: {:.3f}'.format(now_angle),
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv2.imshow(window_name, frame)
    count += 1

    key = cv2.waitKey(delay) & 0xFF
    if  key == ord('q'):
        break
    elif key == ord('a'):
        exp+=1
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    elif key == ord('s'):
        brightness+=1
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    elif key == ord('z'):
        exp-=1
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    elif key == ord('x'):
        brightness-=1
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

cv2.destroyWindow(window_name)