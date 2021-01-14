
import sys
import dlib
from PIL import Image
from imutils import face_utils
import numpy as np
import time
from multiprocessing import RawArray, Value, Process, Array, Pipe
import ctypes
import cv2
import matplotlib.pyplot as plt
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get(raw_array, pipe):
    exp = -6
    brightness = 10
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_FPS, 30)
    X_1 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 500, 500, 3))
    flag = 0
    while True:
        frame = cap.read()[1][134:634, 262:762, :]
        # frame = cv2.resize(frame,(800,600))
        np.copyto(X_1[flag % 100], frame)
        pipe.send_bytes(str(time.time()).encode("utf-8"))
        flag += 1


def calculate_rect(lip):
    r = 5/2
    center_x = int((lip[0] + lip[2] / 2) * r)
    center_y = int((lip[1] + lip[3] / 2) * r)
    overall_h = int(lip[3] * 2.3 * 1.25 * r / 2)
    overall_w = int(lip[2] * 1.8 * 1.25 * r / 2)
    return center_x, center_y, overall_w, overall_h


if __name__ == "__main__":
    (con1, con2) = Pipe()
    raw_array = RawArray(ctypes.c_uint8, 500 * 500 * 3 * 100)
    p = Process(target=get, args=(raw_array, con1))
    p.start()

    window_name = 'frame'

    tm = cv2.TickMeter()
    tm.start()
    delay = 1
    count = 0
    max_count = 10
    fps = 0
    angle = 0
    now_angle = 0
    exflag = 0
    temp = 0
    face_count = 0
    ffps = 0
    X_2 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 500, 500, 3))
    lip_rect = [0, 0, 0, 0]
    angle_list = []
    lost_frame = 0
    s=0
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name,300,0)
    
    while True:
        time_ = float(con2.recv_bytes(18))
        temp += (time.time() - time_)
        frame = X_2[exflag % 100]
        exflag += 1
        image = cv2.cvtColor(cv2.resize(
            frame, (200, 200)), cv2.COLOR_BGR2GRAY)
        rects = DETECTOR(image, 1)
        t1 = time.time()
        for (_, rect) in enumerate(rects):
            shape = PREDICTOR(image, rect)
            shape = face_utils.shape_to_np(shape)
        if rects:
            face_count += 1
            lip = cv2.boundingRect(shape[48:68])
            angle = np.linalg.norm(
                shape[62] - shape[66]) / np.linalg.norm(shape[60] - shape[64])
            angle_list.append(angle)
            if angle > 0.08:
                lip_rect[0], lip_rect[1], lip_rect[2], lip_rect[3] = calculate_rect(
                    lip)
                cv2.rectangle(
                    frame, (lip_rect[0] - lip_rect[2], lip_rect[1] - lip_rect[3]), (lip_rect[0] + lip_rect[2], lip_rect[1] + lip_rect[3]), (0, 0, 255), 2)
        if count == max_count:
            print(temp/max_count, end="\r")
            temp = 0
            tm.stop()
            fps = max_count / tm.getTimeSec()
            ffps = face_count / tm.getTimeSec()
            tm.reset()
            tm.start()
            lost_frame += (count - face_count)
            count = 0
            face_count = 0
            
            now_angle = angle
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'FPS: {:.2f}'.format(fps),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

        cv2.putText(frame, 'FFPS: {:.2f}'.format(ffps),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv2.putText(frame, 'Lost Frames: {:.3f}'.format(lost_frame),
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv2.putText(frame, 'Angle: {:.3f}'.format(now_angle),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        
        cv2.imshow(window_name, frame)
        
        count += 1

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            s+=1
            cv2.moveWindow(window_name,250+(s%3)*600,(s//3)*300)
    x = np.linspace(0, 1, len(angle_list))
    y = np.array(angle_list)
    cv2.destroyWindow(window_name)
    plt.plot(x, y, label="mouth open angle")
    plt.legend()
    plt.show()
    sys.exit()
