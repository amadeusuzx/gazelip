import os
import time

import numpy as np
import cv2
import dlib
from imutils import face_utils

import threading
import queue
import random
import msvcrt
from termcolor import colored


class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
    center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * 4

    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype('float32'))
    i = 0
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    fps = 30
    save_name = f"{path}/{c}/{c}{str(k)}.avi"
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
    for entry in record:
        lip = entry[1]
        new_center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * 4
        if np.linalg.norm(new_center - center) < overall_h/2:
            center = new_center
        frame = entry[0]
        frame = cv2.resize(frame[center[1] - overall_h // 2:center[1] + overall_h // 2,
                                 center[0] - overall_w // 2:center[0] + overall_w // 2], size)
        buffer[i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        i += 1

    video_writer.release()
    print(f"collectd {c} {k}")


if __name__ == "__main__":

    # commands = ['bigger',  # 0
    #             'bubble',  # 1
    #             'drag',  # 2
    #             'drop',  # 3
    #             'last',  # 4
    #             'next',  # 5
    #             'play',  # 6
    #             'smaller',  # 7
    #             'stop',  # 8
    #             'blue',  # 9
    #             'red']  # 10
    origin_commands = ['close_window',
                       'copy_this',
                       'drag',
                       'drop',
                       'enlarge_picture',
                       'fast_forward',
                       'fast_rewind',
                       'paste_here',
                       'pause_video',
                       'play_video',
                       'scroll_down',
                       'scroll_to_bottom',
                       'scroll_up',
                       'select',
                       'speed_down',
                       'speed_up',
                       'translate',
                       'wikipedia']
    commands = random.sample(origin_commands, len(origin_commands))
    # restores the model and optimizer state_dicts

    global detector
    global predictor
    cap = VideoCapture(0)
    path = "./demo_dataset/test"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    buffer = queue.Queue(maxsize=15)
    c = commands.pop(0)
    t1 = 0
    print(f"next command is {c}")
    if not os.path.exists(f"{path}/{c}"):
        os.makedirs(f"{path}/{c}")
    mo = False
    record = []
    j = 0
    k = 35                                                                           
    cleared = False
    while True:
        if cap.q.empty():
            time.sleep(0.01)
        else:
            frame = cap.read()
            image = cv2.cvtColor(cv2.resize(
                frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            if buffer.full():   
                if not cleared:
                    os.system('cls')
                    print(
                        f"_______________{colored(c,'cyan',attrs=['bold'])}_____________________\n\n\n")
                    cleared = True
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
                if cleared:
                    if angle > 0.1:
                        if not mo:
                            print("capturing speech")
                            mo = True
                        t1 = 0
                    elif mo and angle <= 0.1:
                        t1 += 1
                    if t1 > 15 or len(record) == 90:
                        cap.recording = False
                        print(
                            "Record captured! Press Space ␣ to save it, or Z to discard")
                        if msvcrt.getch() == b' ':
                            if len(record) > 40:
                                recognize(record, k, c)
                                j += 1
                                if j == len(origin_commands):
                                    commands = random.sample(
                                        origin_commands, len(origin_commands))
                                    k += 1
                                    j = 0
                                    print(f"\n\nCollected {k} groups. Press Enter key to continue")
                                    while True:
                                        if msvcrt.getch() == b'\r':
                                            break
                                c = commands.pop(0)
                                if not os.path.exists(f"{path}/{c}"):
                                    os.makedirs(f"{path}/{c}")
                                
                            else:
                                print("colored(' Too short, say the command again ','red')")
                        else:
                            print("Record discarded")
                        record = []
                        cleared = False
                        cap.q = queue.Queue(maxsize=100)
                        cap.recording = True
                        mo = False
                        t1 = 0
