import os
import time

import numpy as np
import cv2
import dlib
from imutils import face_utils

import threading
import queue


class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(10, 120  ) # brightness     min: 0   , max: 255 , increment:1  
        self.cap.set(11, 50   ) # contrast       min: 0   , max: 255 , increment:1     
        self.cap.set(12, 70   ) # saturation     min: 0   , max: 255 , increment:1
        self.cap.set(13, 13   ) # hue         
        self.cap.set(14, 50   ) # gain           min: 0   , max: 127 , increment:1
        self.cap.set(15, -3   ) # exposure       min: -7  , max: -1  , increment:1
        self.cap.set(17, 5000 ) # white_balance  min: 4000, max: 7000, increment:1
        self.recording = True
        self.q = queue.Queue(maxsize=100)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):

        while True:

            if self.recording:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.q.put_nowait(frame)
            else:
                time.sleep(0.1)

    def read(self):
        return self.q.get()


def recognize(record, j, c):
    size = (200, 100)

    lip = record[0][1]
    overall_h = int(lip[3] * 2.3) * 5  # *4
    overall_w = int(lip[2] * 1.8) * 5  # *4
    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype('float32'))
    i = 0
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    fps = 30
    save_name = f"{path}/{c}/{c}{str(j)}.avi"
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
    for entry in record:
        lip = entry[1]
        frame = entry[0]
        center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * 4
        frame = cv2.resize(frame[center[1] - overall_h // 2:center[1] + overall_h // 2,
                                 center[0] - overall_w // 2:center[0] + overall_w // 2], size)
        buffer[i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        i += 1

    video_writer.release()
    print(f"collectd {c} {j}")


if __name__ == "__main__":

    commands = ['bigger',  # 0
                'bubble',  # 1
                'drag',  # 2
                'drop',  # 3
                'last',  # 4
                'next',  # 5
                'play',  # 6
                'smaller',  # 7
                'stop',  # 8
                'blue',  # 9
                'red']  # 10
    # restores the model and optimizer state_dicts

    global detector
    global predictor
    cap = VideoCapture(0)
    path = "C:/Users/rkmtlab/Documents/GazeLip/zxsu"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    buffer = queue.Queue(maxsize=15)
    for _ in range(6):
        commands.pop(0)
    c = commands.pop(0)
    t1 = 0
    print(f"next command is {c}")
    if not os.path.exists(f"{path}/{c}"):
        os.makedirs(f"{path}/{c}")
    mo = False
    record = []

    j = 1
    while True:
        if cap.q.empty():
            time.sleep(0.01)
        else:
            frame = cap.read()
            image = cv2.cvtColor(cv2.resize(
                frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            if buffer.full():
                buffer.get_nowait()

            rects = detector(image, 1)
            for (_, rect) in enumerate(rects):
                shape = predictor(image, rect)
                shape = face_utils.shape_to_np(shape)
            if mo:
                while not buffer.empty():
                    record.append(buffer.get())
            if rects:
                lip = cv2.boundingRect(shape[48: 68])
                angle = np.linalg.norm(
                    shape[62] - shape[66]) / np.linalg.norm(shape[60] - shape[64])
                buffer.put_nowait([frame, lip])
                if angle > 0.2:
                    if not mo:
                        print("capturing speech\n")
                    mo = True
                if mo and angle < 0.2:
                    t1 += 1
                if t1 > 15 or len(record) == 100:
                    mo = False
                    t1 = 0
                    if len(record) > 30:
                        cap.q = queue.Queue(maxsize=100)
                        cap.recording = False
                        recognize(record, j, c)
                        cap.recording = True
                        j += 1
                    else:
                        print("too short, please say it again\n")
                    if j > 9:
                        j = 1
                        c = commands.pop(0)
                        print(
                            f"_______________next command is {c}_____________________")
                        os.makedirs(f"{path}/{c}")
                    record = []
