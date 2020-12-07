import os
import time
import queue

import numpy as np
import cv2
import dlib

from imutils import face_utils
from network import R2Plus1DClassifier
import torch
from multiprocessing import Process, RawArray, Value
from WebsocketData import connect_web_socket, send_msg, get_data
import pyautogui
import ctypes


def get(raw_array, flag):
    exp = -6
    brightness = 10
    cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    # cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    # cap.set(cv2.CAP_PROP_FPS, 60)
    X_1 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 600, 800, 3))
    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame,(800,600))
        if flag.value > -1:
            np.copyto(X_1[flag.value % 100], frame)
            flag.value += 1


def recognize(record):
    global CONNECTION
    global DETECTOR
    global PREDICTOR
    global LIP_MODEL
    global COMMANDS
    global FUNC_DICTS
    global CONTEXT
    global TEST
    r = 5

    t1 = time.time()
    # crop image
    size = (60, 30)  # 200*0.6/1.25
    lip = record[0][1]
    overall_h = int(lip[3] * 2.3 * r *0.75)  # 7.5*0.8
    overall_w = int(lip[2] * 1.8 * r *0.75)  #
    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype('float32'))
    count = 0
    for entry in record:
        lip = entry[1]
        center = np.array((lip[0] + lip[2] // 2, lip[1] + lip[3] // 2)) * r
        frame = entry[0]
        frame = cv2.resize(frame[center[1] - overall_h // 2:center[1] + overall_h // 2,
                           center[0] - overall_w // 2:center[0] + overall_w // 2], size)
        # cv2.imshow("window", cv2.resize(frame, (400, 200)))
        # cv2.waitKey(16)
        buffer[count] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        count += 1

    t2 = time.time()
    # print(f"frame processing time:{t2 - t1}s")
    # run model
    buffer = ((buffer - np.mean(buffer)) /
              np.std(buffer)).transpose(3, 0, 1, 2)
    buffer = torch.tensor(np.expand_dims(buffer, axis=0)).cuda()

    outputs = LIP_MODEL(buffer).cpu().detach().numpy()
    t3 = time.time()
    # print(f"inference time:{t3 - t2}s")
    sorted_commands = sorted(list(zip(outputs[0], COMMANDS)))
    print(sorted_commands)
    if not TEST:
        # websocket communication & command execution

        outputs = dict([(COMMANDS[i], outputs[0][i]) for i in range(len(COMMANDS))])
        temp_list = FUNC_DICTS[CONTEXT] + FUNC_DICTS["General"]
        temp_output = [(x, outputs[x]) for x in temp_list]
        command = max(temp_output, key=lambda x: x[1])[0]
        print("------" + command + "------")
        if command == "click":
            pyautogui.click()
        elif command == "maximize":
            pyautogui.press("f")
            send_msg(CONNECTION, command.encode("utf-8"))
            get_data(CONNECTION.recv(8096))
        else:
            send_msg(CONNECTION, command.encode("utf-8"))
            get_data(CONNECTION.recv(8096))

        t4 = time.time()
        # print(f"network communication & command execution time:{t4 - t3}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--test", type=bool,
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
    LIP_MODEL = R2Plus1DClassifier(num_classes=28, layer_sizes=(2, 2, 2, 2, 2, 2))
    state_dicts = torch.load(
        "zxsu3060para12", map_location=torch.device("cuda:0"))
    LIP_MODEL.load_state_dict(state_dicts["state_dict"])
    LIP_MODEL.cuda()
    LIP_MODEL.eval()
    LIP_MODEL(torch.zeros(1, 3, 50, 48, 96, device="cuda:0"))
    print("lip model ready")

    print("camera preparing")
    flag = Value("i",0)
    raw_array = RawArray(ctypes.c_uint8, 800 * 600 * 3 * 100)
    X_2 = np.frombuffer(raw_array, dtype=np.uint8).reshape((100, 600, 800, 3))
    camera_prosess = Process(target=get, args=(raw_array, flag))
    camera_prosess.start()
    print("camera ready")

    buffer = queue.Queue(maxsize=15)
    mouth_open = False
    record = []
    t1 = 0
    exflag = 0
    while True:
        while True:
            if exflag < flag.value:
                break
        frame = X_2[exflag % 100]
        exflag += 1
        image = cv2.resize(frame, (160, 120))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = DETECTOR(image, 1)

        if rects:
            shape = PREDICTOR(image, rects[0])
            np_shape = face_utils.shape_to_np(shape)
            lip = cv2.boundingRect(np_shape[48:68])
            mo_angle = np.linalg.norm(np_shape[62] - np_shape[66]) / np.linalg.norm(np_shape[60] - np_shape[64])
            if not mouth_open:
                if buffer.full():
                    buffer.get_nowait()
                buffer.put_nowait([frame, lip])
                if mo_angle > 0.2:
                    print("capturing speech")
                    mouth_open = True
                    record = list(buffer.queue)
                    buffer = queue.Queue(maxsize=15)
                    if not TEST:
                        send_msg(CONNECTION, "mo".encode('utf-8'))
                        CONTEXT = get_data(CONNECTION.recv(8096))
            else:
                record.append([frame, lip])
                t1 = t1 + 1 if mo_angle < 0.2 else 0

            if t1 > 15 or len(record) == 180:
                print("speech finished")
                if len(record) > 30:
                    flag.value = -1
                    recognize(record)
                if not TEST:
                    send_msg(CONNECTION, "mc".encode('utf-8'))
                flag.value = 0
                exflag = 0
                record = []
                t1 = 0
                mouth_open = False
