import os
import time
import queue

import numpy as np
import cv2
import dlib

from imutils import face_utils
from network import R2Plus1DClassifier
import torch
from torch import nn
from multiprocessing import Process, RawArray, Value, Array
from WebsocketData import connect_web_socket, send_msg, get_data
import pyautogui
import ctypes
import csv

mo_threshold = 0.13
to_threshold = 5
tc_threshold = 30
buffer_size = 35

def get(raw_array, top_flag, stat_flag, lip_rect):
    exp = -6
    brightness = 10
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_FPS, 60)
    X_1 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 500, 500, 3))
    while True:
        frame = cap.read()[1][50:550, 150:650, :]
        # frame_copy = cv2.resize(frame, (500, 500))
        if stat_flag.value:
            np.copyto(X_1[top_flag.value % 100], frame)
            top_flag.value += 1
        #     if stat_flag.value == 2:
        #         cv2.rectangle(
        #             frame_copy, (lip_rect[0] - lip_rect[2], lip_rect[1] - lip_rect[3]), (lip_rect[0] + lip_rect[2], lip_rect[1] + lip_rect[3]), (0, 0, 255), 2)
        # cv2.imshow("window", frame_copy)
        # cv2.waitKey(1)


def calculate_rect(lip):
    r = 5/1.4
    center_x = int((lip[0] + lip[2] / 2) * r)
    center_y = int((lip[1] + lip[3] / 2) * r)
    overall_h = int(lip[3] * 2.3 * r / 2)  # 7.5*0.8
    overall_w = int(lip[2] * 1.8 * r / 2)  #
    return center_x, center_y, overall_w, overall_h


def recognize(record):
    global CONNECTION
    global DETECTOR
    global PREDICTOR
    global LIP_MODEL
    global COMMANDS
    global FUNC_DICTS
    global CONTEXT
    global TEST
    global Recording
    global Recording_count
    r = 5/1.4

    t1 = time.time()
    # crop image
    size = (60, 30)  # 200*0.6/1.25
    lip = record[0][1]
    overall_h = int(lip[3] * 2.3 * r / 2)  #
    overall_w = int(lip[2] * 1.8 * r / 2)  #
    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype('float32'))
    count = 0
    for entry in record:
        lip = entry[1]
        center_x = int((lip[0] + lip[2] / 2) * r)
        center_y = int((lip[1] + lip[3] / 2) * r)
        frame = entry[0]
        frame = cv2.resize(frame[center_y - overall_h:center_y + overall_h,
                                 center_x - overall_w:center_x + overall_w], size)
        cv2.imshow("window", cv2.resize(frame, (400, 200)))
        cv2.waitKey(16)
        buffer[count] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        count += 1

    t2 = time.time()
    print(f"frame processing time:{t2 - t1}s")
    # run model
    buffer = ((buffer - np.mean(buffer)) /
              np.std(buffer)).transpose(3, 0, 1, 2)
    buffer = torch.tensor(np.expand_dims(buffer, axis=0)).cuda()

    outputs = LIP_MODEL(buffer).cpu().detach().numpy()
    t3 = time.time()
    print(f"inference time:{t3 - t2}s")
    sorted_commands = sorted(list(zip(outputs[0], COMMANDS)))
    print(sorted_commands)
    if not TEST:
        # websocket communication & command execution

        outputs = dict([(COMMANDS[i], outputs[0][i])
                        for i in range(len(COMMANDS))])
        temp_list = FUNC_DICTS[CONTEXT] + FUNC_DICTS["General"]
        temp_output = [(x, outputs[x]) for x in temp_list]
        command = max(temp_output, key=lambda x: x[1])[0]
        print("------" + command + "------")
        Recording.append([Recording_count, time.time(),
                          sorted_commands[-1][1], command])
        if command == "click":
            pyautogui.click()
        elif command == "maximize":
            pyautogui.press("f")
            send_msg(CONNECTION, command.encode("utf-8"))
            get_data(CONNECTION.recv(64))
        else:
            send_msg(CONNECTION, command.encode("utf-8"))
            get_data(CONNECTION.recv(64))

        t4 = time.time()
        print(f"network communication & command execution time:{t4 - t3}s")


if __name__ == "__main__":
    Recording = []
    Recording_count = 0
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--subject", type=str,
                        help="subject name and folder name")
    parser.add_argument("--num", type=int, help="start num")
    parser.add_argument("--test", type=bool, default=False,
                        help="whether communicate using websocket")
    args = parser.parse_args()
    TEST = args.test

    FUNC_DICTS = {"General": ["scroll_up", "scroll_down", "go_back", "go_forward", "click"],
                  "SideMenu": ["home", "trending", "subscription", "original", "library"],
                  "NavigationBar": ["profile", "notification", "home"],
                  "Thumbnail": ["play", "watch_later", "add_to_queue"],
                  "MiniPlayer": ["expand", "play", "stop", "previous", "next"],
                  "QueueHead": ["expand", "save"],
                  "Queue": ["delete", "play"],
                  "LikeMenu": ["like", "dislike", "share", "save"],
                  "MainPlayer": ["caption", "play", "stop", "go_back", "go_forward", "previous", "next",
                                 "volume_up", "volume_down", "maximize"]}

    # channelListFuncDict = ["music","gaming","news","movies"]
    COMMANDS = sorted(
        ['caption', 'play', 'stop', 'go_back', 'go_forward', 'previous', 'next', 'volume_up', 'volume_down', 'maximize',
         'expand', 'delete', 'save', 'like', 'dislike', 'share', 'add_to_queue', 'watch_later', 'home', 'trending',
         'subscription', 'original', 'library', 'profile', 'notification', 'scroll_up', 'scroll_down', 'click'])
    print("waiting for ws client...")
    if not TEST:
        CONNECTION = connect_web_socket(10130)
    print("reading face recognition model")
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("recognition model ready")
    print("reading lip model")
    LIP_MODEL = R2Plus1DClassifier(
        num_classes=28, layer_sizes=(2, 2, 2, 2, 2, 2))
    state_dicts = torch.load(
        "xin3060", map_location=torch.device("cuda:0"))
    LIP_MODEL.load_state_dict(state_dicts["state_dict"])
    # load classifier
    # CLASSIFIER = nn.Linear(1024,7)
    # CLASSIFIER.load_state_dict(torch.load("3060training10_classifier7",map_location=torch.device("cuda:0")))
    # CLASSIFIER.cuda()
    # CLASSIFIER.eval()
    LIP_MODEL.cuda()
    LIP_MODEL.eval()

    LIP_MODEL(torch.zeros(1, 3, 50, 48, 96, device="cuda:0"))
    print("lip model ready")

    print("camera preparing")
    top_flag = Value("i", 0)
    stat_flag = Value("i", 1)
    lip_rect = Array('i', [0, 0, 0, 0])
    raw_array = RawArray(ctypes.c_uint8, 500 * 500 * 3 * 100)
    X_2 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 500, 500, 3))
    camera_process = Process(target=get, args=(
        raw_array, top_flag, stat_flag, lip_rect))
    camera_process.start()
    print("camera ready")

    buffer = queue.Queue(maxsize=buffer_size)
    mouth_open = False
    record = []
    t_close = 0
    t_open = 0
    exflag = 0

    try:
        while True:
            while True:
                if exflag < top_flag.value:
                    break
            frame = X_2[exflag % 100]
            exflag += 1
            image = cv2.resize(frame, (140, 140))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = DETECTOR(image, 1)
            if buffer.full():
                buffer.get_nowait()
            if rects:
                shape = PREDICTOR(image, rects[0])
                np_shape = face_utils.shape_to_np(shape)
                lip = cv2.boundingRect(np_shape[48:68])
                mo_angle = np.linalg.norm(
                    np_shape[62] - np_shape[66]) / np.linalg.norm(np_shape[60] - np_shape[64])
                if not mouth_open:
                    buffer.put_nowait([frame, lip])
                    if mo_angle > mo_threshold:
                        t_open += 1
                        if t_open == to_threshold:
                            t_open = 0
                            print("capturing speech")
                            lip_rect[0], lip_rect[1], lip_rect[2], lip_rect[3] = calculate_rect(
                                lip)
                            # stat_flag.value = 2
                            mouth_open = True
                            record = list(buffer.queue)
                            buffer = queue.Queue(maxsize=buffer_size)
                            if not TEST:
                                send_msg(CONNECTION, "mo".encode('utf-8'))
                                CONTEXT = get_data(CONNECTION.recv(64))
                    else:
                        t_open = 0
                else:
                    lip_rect[0], lip_rect[1], _, _ = calculate_rect(lip)
                    record.append([frame, lip])
                    t_close = t_close + 1 if mo_angle < mo_threshold else 0

                if t_close > tc_threshold or len(record) == 180:
                    Recording_count += 1
                    stat_flag.value = 0
                    print("speech finished")
                    if len(record) > buffer_size+tc_threshold+5:
                        recognize(record)
                    else:
                        print(len(record))
                        Recording.append(
                            [str(Recording_count), time.time(), "ignored"])
                    if not TEST:
                        send_msg(CONNECTION, "mc".encode('utf-8'))
                    record = []
                    cleared = False
                    exflag = 0
                    top_flag.value = 0
                    stat_flag.value = 1
                    mouth_open = False
                    t_close = 0
    except KeyboardInterrupt:
        camera_process.join()
        with open(f"{args.subject}-{args.num}.csv", "w") as csv_file:
            wr = csv.writer(csv_file, dialect='excel')
            wr.writerows(Recording)
