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
from multiprocessing import Process, Queue, Value


def get(q, recording):
    camera_id = 0

    exp = -6
    brightness = 10
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_FPS, 60)
    while True:
        _, frame = cap.read()
        if recording:
            q.put(frame)


def recognize(record, j, c):
    size = (200, 100)
    r = 5
    lip = record[0][1]
    overall_h = lip[3] * 2.3 * 1.25 * r  # *5
    overall_w = lip[2] * 1.8 * 1.25 * r  # *5

    center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * r

    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype("float32"))
    i = 0
    fourcc = cv2.VideoWriter_fourcc(*"I420")
    fps = 30
    save_name = f"{path}/{c}/{c}{str(k)}.avi"
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
    for entry in record:
        lip = entry[1]
        new_center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * r
        if np.linalg.norm(new_center - center) < overall_h/2:
            center = new_center
        frame = entry[0]
        frame = cv2.resize(frame[int(center[1] - overall_h/2):int(center[1] + overall_h / 2),
                                 int(center[0] - overall_w / 2):int(center[0] + overall_w / 2)], size)
        buffer[i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        i += 1

    video_writer.release()
    print(f"collected {c} {k}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--subject", type=str,
                        help="subject name and folder name")
    parser.add_argument("--num", type=int, help="start num")

    args = parser.parse_args()
    if not args.subject:
        print("input parameters!")
        sys.exit()

    origin_commands = [
        'caption',
        'play',
        'stop',
        'go_back',
        'go_forward',
        'previous',
        'next',
        'turn_it_up',
        'turn_it_down',
        'full_screen',
        'expand',
        'delete',
        'save',
        'like',
        'dislike',
        'share',
        'add_to_queue',
        'watch_later',
        'home',
        'trending',
        'subscriptions',
        'originals',
        'library',
        'voice_search',
        'profile',
        'notifications',
        'scroll_up',
        'scroll_down',
        'click']
    commands = random.sample(origin_commands, len(origin_commands))
    # restores the model and optimizer state_dicts

    mp_queue = Queue(maxsize=100)
    recording = Value('i', 1)
    p1 = Process(target=get, args=(mp_queue, recording,))
    p1.start()

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
        frame = mp_queue.get()
        image = cv2.cvtColor(cv2.resize(
            frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        if buffer.full():
            if not cleared:
                os.system("cls")
                print(
                    f"\n\n\n\n\n\n                        {colored(c,'cyan',attrs=['bold'])}              \n\n\n")
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
                if cleared and mo_angle > 0.15:
                    print("capturing speech")
                    mouth_open = True
                    record = list(buffer.queue)
                    buffer = queue.Queue(maxsize=15)
            else:
                record.append([frame, lip])
                t1 = t1+1 if mo_angle < 0.1 else 0
            if t1 > 25 or len(record) == 180:
                print(f"collected {len(record)} frames")
                recording = 0
                if len(record) <= 50:
                    print(f"{colored(' Too short, say the command again ','red')}")
                else:
                    print(
                        "Record captured! Press Space ␣ to save it, or any other key to discard")
                    if msvcrt.getch() == b" ":
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
                                if msvcrt.getch() == b"\r":
                                    break
                        c = commands.pop(0)
                        if not os.path.exists(f"{path}/{c}"):
                            os.makedirs(f"{path}/{c}")
                    else:
                        print("Record discarded")

                record = []
                cleared = False
                qsize = mp_queue.qsize()
                for _ in range(qsize):
                    mp_queue.get()
                recording = 1
                mouth_open = False
                t1 = 0
