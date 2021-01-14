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
import time

mo_threshold = 0.1
to_threshold = 5
tc_threshold = 30
buffer_size = 15
block_size = 1
face_recognition_size = 120


def get(raw_array, top_flag, stat_flag, lip_rect, i, block_finished):
    origin_commands = ['caption', 'play', 'stop', 'go_back', 'go_forward', 'previous', 'next', 'volume_up', 'volume_down', 'full_screen', 'expand', 'delete', 'save', 'like',
                       'dislike', 'share', 'add_to_queue', 'watch_later', 'homepage', 'trending', 'subscription', 'original', 'library', 'profile', 'notification', 'scroll_up', 'scroll_down']
    exp = -6
    brightness = 10
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_FPS, 30)
    X_1 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 768, 1024, 3))
    k = block_finished.value
    cv2.namedWindow('window')
    cv2.moveWindow('window', 250+(k % 3)*600, (k//3)*300)
    while True:
        frame = cap.read()[1]
        frame_copy = cv2.resize(frame, (500, 500))
        if stat_flag.value > 0:
            np.copyto(X_1[top_flag.value % 100], frame)
            top_flag.value += 1
            if stat_flag.value == 2:
                cv2.rectangle(
                    frame_copy, (lip_rect[0] - lip_rect[2], lip_rect[1] - lip_rect[3]), (lip_rect[0] + lip_rect[2], lip_rect[1] + lip_rect[3]), (0, 0, 255), 2)
            frame_copy = cv2.flip(cv2.resize(frame_copy,(500,500), 1)
            cv2.putText(frame_copy, origin_commands[i.value].replace("_", " "), (175, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 160, 0), 2, cv2.LINE_AA)
        elif stat_flag.value == 0:
            frame_copy = cv2.flip(cv2.resize(frame_copy,(500,500), 1)
            cv2.putText(frame_copy, "press space key to continue", (50, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_copy, "press 'Q' to discard", (50, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            if block_finished.value != k:
                k = block_finished.value
                cv2.moveWindow('window', 250+(k % 3)*600, (k//3)*300)
        elif stat_flag.value == -1:
            frame_copy = cv2.flip(frame_copy, 1)
            cv2.putText(frame_copy, "speech is to short. try slower agiain", (50, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("window", frame_copy)
        cv2.waitKey(1)


def calculate_rect(lip):
    r = 500/160
    center_x = int((lip[0] + lip[2] / 2) * r)
    center_y = int((lip[1] + lip[3] / 2) * r)
    overall_h = int(lip[3] * 2.91 * r / 2)  # 2.3*1.25
    overall_w = int(lip[2] * 2.4 * r / 2)  # 1.8 *1.25
    return center_x, center_y, overall_w, overall_h


def recognize(record, j, c):
    r = 500/160
    size = (120, 60)
    lip = record[0][1]
    overall_h = int(lip[3] * 2.91 * r / 2)
    overall_w = int(lip[2] * 2.4 * r / 2)
    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype("float32"))
    i = 0
    fourcc = cv2.VideoWriter_fourcc(*"I420")
    fps = 30
    save_name = f"{path}/{c}/{c}{str(k)}.avi"
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
    for entry in record:
        lip = entry[1]
        center_x = int((lip[0] + lip[2] / 2) * r)
        center_y = int((lip[1] + lip[3] / 2) * r)
        frame = entry[0]
        frame = cv2.resize(frame[center_y - overall_h:center_y + overall_h,
                                 center_x - overall_w:center_x + overall_w], size)
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
    if args.subject == "test":
        block_size = 1
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
        'full_screen',
        'expand',
        'delete',
        'save',
        'like',
        'dislike',
        'share',
        'add_to_queue',
        'watch_later',
        'homepage',
        'trending',
        'subscription',
        'original',
        'library',
        'profile',
        'notification',
        'scroll_up',
        'scroll_down']
    commands = random.sample(origin_commands, len(origin_commands))

    # multiprocessing camera
    k = args.num
    block_finished = Value("i", k-1)
    command_index = Value("i", 0)
    top_flag = Value("i", 0)
    stat_flag = Value("i", 1)
    lip_rect = Array('i', [0, 0, 0, 0])
    raw_array = RawArray(ctypes.c_uint8, 500 * 500 * 3 * 100)
    X_2 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 500, 500, 3))
    camera_process = Process(target=get, args=(
        raw_array, top_flag, stat_flag, lip_rect, command_index, block_finished))
    camera_process.start()

    # dlib model loading
    path = "H:/Gaze-Lip-Data/GazeLipDatasets/" + args.subject
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    buffer = queue.Queue(maxsize=buffer_size)
    c = commands.pop(0)
    command_index.value = origin_commands.index(c)
    t_close = 0
    t_open = 0
    print(f"next command is {c}")
    if not os.path.exists(f"{path}/{c}"):
        os.makedirs(f"{path}/{c}")
    mouth_open = False
    record = []
    j = 0

    cleared = False
    exflag = 0

    while True:
        while True:
            if exflag < top_flag.value:
                break
        frame = X_2[exflag % 100]
        exflag += 1
        image = cv2.cvtColor(cv2.resize(
            frame, (160, 160)), cv2.COLOR_BGR2GRAY)
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
            mo_angle = np.linalg.norm(
                np_shape[62] - np_shape[66]) / np.linalg.norm(np_shape[60] - np_shape[64])
            if not mouth_open:
                buffer.put_nowait([frame, lip])
                if cleared and mo_angle > mo_threshold:
                    t_open += 1
                    if t_open > to_threshold:
                        lip_rect[0], lip_rect[1], lip_rect[2], lip_rect[3] = calculate_rect(
                            lip)
                        print("capturing speech")
                        mouth_open = True
                        stat_flag.value = 2
                        record = list(buffer.queue)
                        buffer = queue.Queue(maxsize=buffer_size)
                        t_open = 0
                else:
                    t_open = 0
            else:
                lip_rect[0], lip_rect[1], _, _ = calculate_rect(lip)
                record.append([frame, lip])
                t_close = t_close + 1 if mo_angle < mo_threshold else 0
            if t_close > tc_threshold or len(record) == 180:
                print(f"collected {len(record)} frames")
                if len(record) <= tc_threshold+buffer_size+5:
                    stat_flag.value = -1
                    time.sleep(0.5)
                else:
                    stat_flag.value = 0
                    while msvcrt.kbhit():
                        msvcrt.getch()
                    print(
                        "Record captured! Press Space ␣ to save it, or press 'Q' to discard")
                    if msvcrt.getch() == b" ":

                        recognize(record, k, c)
                        j += 1
                        if j == len(origin_commands):
                            j = 0
                            commands = random.sample(
                                origin_commands, len(origin_commands))
                            if k % block_size == 0:
                                if k == 9:
                                    break
                                block_finished.value = k
                                print(
                                    f"\n\nCollected {k} groups. Press Enter key twice to continue")
                                while True:
                                    if msvcrt.getch() == b"\r" and msvcrt.getch() == b"\r":
                                        break

                            k += 1
                        c = commands.pop(0)
                        command_index.value = origin_commands.index(c)
                        if not os.path.exists(f"{path}/{c}"):
                            os.makedirs(f"{path}/{c}")
                    else:
                        print("Record discarded")

                record = []
                cleared = False
                exflag = 0
                top_flag.value = 0
                stat_flag.value = 1
                mouth_open = False
                t_close = 0
