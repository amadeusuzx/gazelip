import time
import queue

import numpy as np
import cv2
import dlib
import math 

from imutils import face_utils
from network import R2Plus1DClassifier
import torch

from multiprocessing import Process, RawArray, Value, Array
from WebsocketData import connect_web_socket, send_msg, get_data
import pyautogui    
import ctypes
import csv
import datetime
# import random
import subprocess
import socket
from threading import Thread

mo_threshold = 0.1
to_threshold = 5
tc_threshold = 15
buffer_size = 15

face_recognition_size = 120
pyautogui.FAILSAFE = False


def socket_thread():
    global record
    global exflag
    global top_flag
    global stat_flag
    global mouth_open
    global t_close
    global LipReading_flag

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 50007))
        s.listen(1)
        gaze = None
        while True:
            conn, _ = s.accept()
            print("console connected")
            with conn:
                while True:
                    data = conn.recv(1024)
                    if data == b"g":
                        LipReading_flag = True
                        print("--------lip started-----------")
                        gaze = subprocess.Popen("GazeCursor.exe")
                        print("--------gaze started-----------")
                    elif data == b"s":
                        LipReading_flag = False
                        record = []
                        # record = []
                        # exflag = 0
                        # top_flag.value = 0
                        # stat_flag.value = 1
                        # mouth_open = False
                        # t_close = 0
                        print("--------lip stoped--------")
                        if gaze:
                            gaze.kill()
                            print("--------gaze stoped--------")


def get(raw_array, top_flag, stat_flag, lip_rect):
    # size = (500,500)
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # fps = 60
    # save_name = f"H:/Gaze-Lip-Data/record{random.randint(1,10000)}.avi"
    # video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)

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
        if stat_flag.value:
            np.copyto(X_1[top_flag.value % 100], frame)
            top_flag.value += 1

        if stat_flag.value == 2:
            frame_copy = np.copy(frame)
            cv2.rectangle(frame_copy, (lip_rect[0] - lip_rect[2], lip_rect[1] - lip_rect[3]),
                          (lip_rect[0] + lip_rect[2], lip_rect[1] + lip_rect[3]), (0, 0, 255), 2)
            cv2.imshow("window", frame_copy)
        else:
            cv2.imshow("window", frame)
        cv2.waitKey(1)
        # video_writer.write(frame)


def calculate_rect(lip):
    r = 500/face_recognition_size
    center_x = int((lip[0] + lip[2] / 2) * r)
    center_y = int((lip[1] + lip[3] / 2) * r)
    overall_h = int(lip[3] * 1.94 * r / 2)  # 2.3*1.25
    overall_w = int(lip[2] * 1.6 * r / 2)  # 1.8 *1.25
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

    r = 500/face_recognition_size

    # crop image
    size = (80, 40)  # 200*0.6/1.25
    lip = record[0][1]
    overall_h = int(lip[3] * 1.94 * r / 2)  # r = 2.91/1.5
    overall_w = int(lip[2] * 1.6 * r / 2)  # 1.6 = 2.4/1.5
    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype('float32'))
    count = 0
    for entry in record:
        lip = entry[1]
        center_x = int((lip[0] + lip[2] / 2) * r)
        center_y = int((lip[1] + lip[3] / 2) * r)
        frame = entry[0]
        frame = cv2.resize(frame[center_y - overall_h:center_y + overall_h,
                                    center_x - overall_w:center_x + overall_w], size)
        # cv2.imshow("window", cv2.resize(frame, (400, 200)))
        # cv2.waitKey(16)
        buffer[count] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        count += 1

    # run model
    buffer = ((buffer - np.mean(buffer)) /
              np.std(buffer)).transpose(3, 0, 1, 2)
    buffer = torch.tensor(np.expand_dims(buffer, axis=0)).cuda()

    outputs = LIP_MODEL(buffer).cpu().detach().numpy()
    probabilities = [math.exp(x) for x in outputs[0]]
    exp_sum = sum(probabilities)
    probabilities = np.array(probabilities)/exp_sum
    sorted_commands = sorted(list(zip(probabilities, COMMANDS)))
    print(sorted_commands)
    print("raw command: " + sorted_commands[-1][-1])
    if not TEST:
        # websocket communication & command execution

        outputs = dict([(COMMANDS[i], probabilities)
                        for i in range(len(COMMANDS))])
        temp_list = FUNC_DICTS[CONTEXT]
        temp_output = [(x, outputs[x]) for x in temp_list]
        command = max(temp_output, key=lambda x: x[1])[0]
        print("squeezed command: " + command)
        Recording.append(
            [time.time()-START_TIME, sorted_commands[-1][1], command])
        if command == "full_screen":
            pyautogui.press("f")
            send_msg(CONNECTION, command.encode("utf-8"))
        else:
            send_msg(CONNECTION, command.encode("utf-8"))


if __name__ == "__main__":
    Recording = []
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--subject", type=str,default="test",
                        help="subject name and folder name")
    parser.add_argument("--name", type=str, default="test", help="start name")
    parser.add_argument("--test", type=int, default=0,
                        help="whether communicate using websocket")
    args = parser.parse_args()
    TEST = args.test
    model_path = "H:/Gaze-Lip-Data/GazeLipModels/"+args.subject+".pt"

    FUNC_DICTS = {"SideMenu": ["homepage", "trending", "subscription", "original", "library"],
                  "NavigationBar": ["profile", "notification", "homepage","scroll_up", "scroll_down", "go_back", "go_forward"],
                  "Thumbnail": ["play", "watch_later", "add_to_queue"],
                  "MiniPlayer": ["expand", "play", "stop", "previous", "next"],
                  "QueueHead": ["expand", "save"],
                  "Queue": ["delete", "play"],
                  "LikeMenu": ["like", "dislike", "share", "save"],
                  "MainPlayer": ["caption", "play", "stop", "go_back", "go_forward", "previous", "next",
                                 "volume_up", "volume_down", "full_screen"]}

    # channelListFuncDict = ["music","gaming","news","movies"]
    COMMANDS = sorted(
        ['caption', 'play', 'stop', 'go_back', 'go_forward', 'previous', 'next', 'volume_up', 'volume_down', 'full_screen',
         'expand', 'delete', 'save', 'like', 'dislike', 'share', 'add_to_queue', 'watch_later', 'homepage', 'trending',
         'subscription', 'original', 'library', 'profile', 'notification', 'scroll_up', 'scroll_down'])
    print("waiting for ws client...")
    if not TEST:
        CONNECTION = connect_web_socket(10130)
        CONNECTION.settimeout(1)
    print("reading face recognition model")
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("recognition model ready")
    print("reading lip model")
    LIP_MODEL = R2Plus1DClassifier(
        num_classes=27, layer_sizes=(2, 2, 2, 2, 2, 2))
    state_dicts = torch.load(
        model_path, map_location=torch.device("cuda:0"))
    LIP_MODEL.load_state_dict(state_dicts["state_dict"])
    LIP_MODEL.cuda()
    LIP_MODEL.eval()

    LIP_MODEL(torch.zeros(1, 3, 50, 48, 96, device="cuda:0"))
    print("lip model ready")
    LipReading_flag = False

    st = Thread(target=socket_thread, args=())
    st.setDaemon(True)
    st.start()

    print("camera preparing")
    lip_rect = Array('i', [0, 0, 0, 0])
    top_flag = Value("i", 0)
    stat_flag = Value("i", 1)
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

    START_TIME = time.time()
    START_DATETIME = datetime.datetime.now()
    Recording.append([START_DATETIME, "record"])

    try:
        while True:
            while True:
                if exflag < top_flag.value:
                    break
            frame = X_2[exflag % 100]
            exflag += 1
            image = cv2.resize(
                frame, (face_recognition_size, face_recognition_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = DETECTOR(image, 1)
            if buffer.full():
                buffer.get_nowait()
            if rects:
                shape = PREDICTOR(image, rects[0])
                np_shape = face_utils.shape_to_np(shape)
                lip = cv2.boundingRect(np_shape[48:68])
                mo_angle = np.linalg.norm(np_shape[62] - np_shape[66]) / np.linalg.norm(
                    np_shape[60] - np_shape[64]) if (LipReading_flag or TEST) else 0
                if not mouth_open:
                    buffer.put_nowait([frame, lip])
                    if mo_angle > mo_threshold:
                        t_open += 1
                        if t_open > to_threshold:
                            if not TEST:
                                send_msg(CONNECTION, "mo".encode('utf-8'))
                                CONTEXT = get_data(CONNECTION.recv(64))
                                if CONTEXT == "null":
                                    t_open = 0
                                    continue
                            lip_rect[0], lip_rect[1], lip_rect[2], lip_rect[3] = calculate_rect(
                                lip)
                            t_open = 0
                            stat_flag.value = 2
                            mouth_open = True
                            record = list(buffer.queue)
                            buffer = queue.Queue(maxsize=buffer_size)
                            Recording.append(
                                [time.time()-START_TIME, "mouth open"])

                    else:
                        t_open = 0
                else:
                    lip_rect[0], lip_rect[1], _, _ = calculate_rect(lip)
                    record.append([frame, lip])
                    t_close = t_close + 1 if mo_angle < mo_threshold else 0

                if t_close > tc_threshold or len(record) == 180:
                    stat_flag.value = 0
                    print(f"{len(record)} frames")
                    # if len(record) > buffer_size+tc_threshold:
                    #                         else:
                    #     Recording.append(
                    #         [time.time()-START_TIME, "ignored"])
                    recognize(record)

                    Recording.append([time.time()-START_TIME, "mouth closed"])
                    record = []
                    exflag = 0
                    top_flag.value = 0
                    stat_flag.value = 1
                    mouth_open = False
                    t_close = 0
    except KeyboardInterrupt:
        camera_process.join()
        with open(f"{args.subject}-{args.name}.csv", "w") as csv_file:
            wr = csv.writer(csv_file, dialect='excel')
            wr.writerows(Recording)
