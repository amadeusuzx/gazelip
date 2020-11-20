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
import sys


class VideoCapture:

    def __init__(self, capture):
        self.cap = capture
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
    size = (120,72)
    r = 5
    lip = record[0][1]
    overall_h = int(lip[3] * 2.3 * r *1.25)  # *5
    overall_w = int(lip[2] * 1.8 * r *1.25)  # *5
    print(overall_h)
    center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * r

    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype('float32'))
    i = 0
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    fps = 30
    save_name = f"{path}/{c}/{c}{str(k)}.avi"
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
    for entry in record:
        lip = entry[1]
        new_center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * r
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
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--subject', type=str,
                        help='subject name and folder name')
    parser.add_argument('--num', type=int, help='start num')

    args = parser.parse_args()
    if not args.subject:
        print("input parameters!")
        sys.exit()

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
    # origin_commands = ['close_window',
    #                    'copy_this',
    #                    'drag',
    #                    'drop',
    #                    'enlarge_picture',
    #                    'fast_forward',
    #                    'fast_rewind',
    #                    'paste_here',
    #                    'pause_video',
    #                    'play_video',
    #                    'scroll_down',
    #                    'scroll_to_bottom',
    #                    'scroll_up',
    #                    'select',
    #                    'speed_down',
    #                    'speed_up',
    #                    'translate',
    #                    'wikipedia']
    # origin_commands = [#'close_window',
    #                 'copy',
    #                 'drag',
    #                 'drop',
    #                 'enlarge',
    #                 'close',
    #                 'open',
    #                 'forward',
    #                 'rewind',
    #                 'paste',
    #                 'pause',
    #                 'play',
    #                 'down',
    #                 'up',
    #                 'select',
    #                 'fast',
    #                 'slow',
    #                 'translate',
    #                 'wikipedia',
    #                 "google"]

    commands = random.sample(origin_commands, len(origin_commands))
    # restores the model and optimizer state_dicts
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    capture.set(cv2.CAP_PROP_FPS, 60)
    capture.read()
    cap = VideoCapture(capture)

    path = "./user_study/"+args.subject
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    buffer = queue.Queue(maxsize=15)
    c = commands.pop(0)
    t1 = 0
    print(f"next command is {c}")
    if not os.path.exists(f"{path}/{c}"):
        os.makedirs(f"{path}/{c}")
    mouth_open = False
    record = []
    j = 0
    k = args.num
    cleared = False
    while True:

        frame = cap.read()
        image = cv2.cvtColor(cv2.resize(
            frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        if buffer.full():
            if not cleared:
                os.system('cls')
                print(
                    f"_______________{colored(c,'cyan',attrs=['bold'])}_____________________\n\n\n")
                cleared = True
            buffer.get_nowait()

        rects = DETECTOR(image, 1)
        if rects:
            shape = PREDICTOR(image, rects[0])
            np_shape = face_utils.shape_to_np(shape)
            lip = cv2.boundingRect(np_shape[48:68])
            mo_angle = np.linalg.norm(
                np_shape[62] - np_shape[66]) / np.linalg.norm(np_shape[60] - np_shape[64])
            if not mouth_open:
                buffer.put_nowait([frame, lip])
                if cleared and mo_angle > 0.1:
                    print("capturing speech")
                    mouth_open = True
                    record = list(buffer.queue)
                    buffer = queue.Queue(maxsize=15)
            else:
                record.append([frame, lip])
                t1 = t1+1 if mo_angle < 0.1 else 0
            if t1 > 25 or len(record) == 180:
                cap.recording = False
                if len(record) <= 50:
                    print(f"{colored(' Too short, say the command again ','red')}")
                else:
                    print(
                        "Record captured! Press Space ␣ to save it, or any other key to discard")
                    if msvcrt.getch() == b' ':
                        recognize(record, k, c)
                        j += 1
                        if j == len(origin_commands):
                            commands = random.sample(
                                origin_commands, len(origin_commands))
                            k += 1
                            if k % 10 == 0:
                                print(
                                    f"\n\nCollected {k} groups. Take a little break!")
                                sys.exit()
                            j = 0
                            print(
                                f"\n\nCollected {k} groups. Press Enter key to continue")
                            while True:
                                if msvcrt.getch() == b'\r':
                                    break
                        c = commands.pop(0)
                        if not os.path.exists(f"{path}/{c}"):
                            os.makedirs(f"{path}/{c}")
                    else:
                        print("Record discarded")

                record = []
                cleared = False
                cap.q = queue.Queue(maxsize=100)
                cap.recording = True
                mouth_open = False
                t1 = 0
