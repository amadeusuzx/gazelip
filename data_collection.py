import os
import ctypes
import sys
from multiprocessing import Process, Value, RawArray, Array

import queue
import random
import msvcrt
from termcolor import colored

import numpy as np
import cv2
import dlib
from imutils import face_utils

def get(raw_array, top_flag, stat_flag, lip_rect):
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
        _, frame = cap.read()
        frame_copy = cv2.resize(frame, (400, 300))
        if stat_flag.value:
            np.copyto(X_1[top_flag.value % 100], frame)
            top_flag.value += 1
            if stat_flag.value == 2:
                cv2.rectangle(frame_copy, (lip_rect[0], lip_rect[1]), (lip_rect[2], lip_rect[3]), (0, 0, 255), 2)
        cv2.imshow("window", frame_copy)
        cv2.waitKey(1)


def calculate_rect(lip):
    r = 2.5
    x1 = int((lip[0] - lip[2] * 0.625) * r)
    y1 = int((lip[1] - lip[3] * 0.9375) * r)
    x2 = x1 + int(lip[2] * 1.8 * 1.25 * r)  # *5
    y2 = y1 + int(lip[3] * 2.3 * 1.25 * r)  # *5
    return x1, y1, x2, y2


def recognize(record, j, c):
    r = 5
    size = (200, 100)
    lip = record[0][1]
    overall_h = int(lip[3] * 2.3 * 1.25 * r)  # 7.5*0.8
    overall_w = int(lip[2] * 1.8 * 1.25 * r)  #
    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype("float32"))
    i = 0
    fourcc = cv2.VideoWriter_fourcc(*"I420")
    fps = 30
    save_name = f"{path}/{c}/{c}{str(k)}.avi"
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
    for entry in record:
        lip = entry[1]
        center = np.array((lip[0] + lip[2] // 2, lip[1] + lip[3] // 2)) * r
        frame = entry[0]
        frame = cv2.resize(frame[center[1] - overall_h // 2:center[1] + overall_h // 2,
                                 center[0] - overall_w // 2:center[0] + overall_w // 2], size)
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
        'volume_up',
        'volume_down',
        'maximize',
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
        'subscription',
        'original',
        'library',
        'profile',
        'notification',
        'scroll_up',
        'scroll_down',
        'click']
    commands = random.sample(origin_commands, len(origin_commands))
    
    top_flag = Value("i", 0)
    stat_flag = Value("i", 1)
    lip_rect = Array('i', [0, 0, 0, 0])
    raw_array = RawArray(ctypes.c_uint8, 800 * 600 * 3 * 100)
    X_2 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 600, 800, 3))
    camera_process = Process(target=get, args=(raw_array, top_flag, stat_flag, lip_rect))
    camera_process.start()

    path = "H:/GazeLipDatasets/" + args.subject
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    buffer = queue.Queue(maxsize=25)
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
    exflag = 0

    while True:
        while True:
            if exflag < top_flag.value:
                break
        frame = X_2[exflag % 100]
        exflag += 1
        image = cv2.cvtColor(cv2.resize(
            frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        if buffer.full():
            if not cleared:
                os.system("cls")
                print(
                    f"\n\n\n\n\n\n                               {colored(c, 'cyan', attrs=['bold'])}\n\n\n")
                cleared = True
            buffer.get_nowait()

        rects = DETECTOR(image, 1)
        if rects:
            shape = PREDICTOR(image, rects[0])
            np_shape = face_utils.shape_to_np(shape)
            lip = cv2.boundingRect(np_shape[48:68])
            lip_rect[0], lip_rect[1], lip_rect[2], lip_rect[3] = calculate_rect(
                lip)
            mo_angle = np.linalg.norm(
                np_shape[62] - np_shape[66]) / np.linalg.norm(np_shape[60] - np_shape[64])
            if not mouth_open:
                buffer.put_nowait([frame, lip])
                if cleared and mo_angle > 0.1:
                    print("capturing speech")
                    mouth_open = True
                    stat_flag.value = 2
                    record = list(buffer.queue)
                    buffer = queue.Queue(maxsize=25)
            else:
                record.append([frame, lip])
                t1 = t1 + 1 if mo_angle < 0.1 else 0
            if t1 > 25 or len(record) == 180:
                print(f"collected {len(record)} frames")
                stat_flag.value = 0
                if len(record) <= 65:
                    print(f"{colored(' Too short, say the command again ', 'red')}")
                else:
                    print(
                        "Record captured! Press Space ␣ to save it, or any other key to discard")
                    if msvcrt.getch() == b" ":
                        recognize(record, k, c)
                        j += 1
                        if j == len(origin_commands):
                            j = 0
                            commands = random.sample(
                                origin_commands, len(origin_commands))
                            if k % 5 == 0:
                                print(
                                    f"\n\nCollected {k} groups. Press Enter key twice to continue")
                                while True:
                                    if msvcrt.getch() == b"\r" and msvcrt.getch() == b"\r":
                                        break

                            k += 1
                        c = commands.pop(0)
                        if not os.path.exists(f"{path}/{c}"):
                            os.makedirs(f"{path}/{c}")
                    else:
                        print("Record discarded")

                record = []
                cleared = False
                stat_flag.value = 1
                top_flag.value = 0
                exflag = 0
                mouth_open = False
                t1 = 0
