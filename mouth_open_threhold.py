import cv2
import sys
import dlib
from PIL import Image
from imutils import face_utils
import numpy as np
import time
from multiprocessing import RawArray, Value, Process, Array, Pipe
import ctypes

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get(raw_array, flag):
    exp = -6
    brightness = 10
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_FPS, 60)
    X_1 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 600, 800, 3))
    while True:
        ret, frame = cap.read()
        np.copyto(X_1[flag % 100], frame)
        pipe.send_bytes(str(time.time()).encode("utf-8"))
        flag+=1


if __name__ == "__main__":
    (con1, con2) = Pipe()
    raw_array = RawArray(ctypes.c_uint8, 800 * 600 * 3 * 100)
    p = Process(target=get, args=(raw_array, con1))
    p.start()

    window_name = 'frame'

    tm = cv2.TickMeter()
    tm.start()
    delay = 1
    count = 0
    max_count = 20
    fps = 0
    angle = 0
    now_angle = 0
    exflag = 0
    temp = 0
    X_2 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 600, 800, 3))
    while True:
        time_ = float(con2.recv_bytes(18))
        temp+=(time.time() - time_)
        frame = X_2[exflag % 100 ]
        exflag += 1
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
            print(temp/max_count,end="\r")
            temp = 0
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
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    sys.exit()
