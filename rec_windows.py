import os
import time
import requests
from io import BytesIO
import win32clipboard

import threading
import queue
import socket
import base64
import hashlib

import numpy as np
import cv2
import dlib
from PIL import Image
from imutils import face_utils
from network import R2Plus1DClassifier
import torch

import pyautogui
import pyperclip
from pygame import mixer


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
                    print("camera faliure")
                    break
                self.q.put_nowait(frame)
            else:
                time.sleep(0.1)

    def read(self):
        return self.q.get()

def send_to_clipboard(clip_type, data):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()

def copy_image(url):

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    output = BytesIO()
    img.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()
    send_to_clipboard(win32clipboard.CF_DIB, data)

def get_data(info):
    payload_len = info[1] & 127
    if payload_len == 126:
        mask = info[4:8]
        decoded = info[8:]
    elif payload_len == 127:
        mask = info[10:14]
        decoded = info[14:]
    else:
        mask = info[2:6]
        decoded = info[6:]

    bytes_list = bytearray() 
    for i in range(len(decoded)):
        chunk = decoded[i] ^ mask[i % 4]
        bytes_list.append(chunk)
    try:
        body = str(bytes_list, encoding='utf-8')
    except:
        print(f"bytes_list is : {bytes_list}")
        return "invalid"
    return body

def get_headers(data):

    header_dict = {}
    data = str(data, encoding='utf-8')

    header, _ = data.split('\r\n\r\n', 1)
    header_list = header.split('\r\n')
    for i in range(0, len(header_list)):
        if i == 0:
            if len(header_list[i].split(' ')) == 3:
                header_dict['method'], header_dict['url'], header_dict['protocol'] = header_list[i].split(
                    ' ')
        else:
            k, v = header_list[i].split(':', 1)
            header_dict[k] = v.strip()
    return header_dict


def send_msg(conn, msg_bytes):                      
    
    import struct

    token = b"\x81"
    length = len(msg_bytes)
    if length < 126:
        token += struct.pack("B", length)
    elif length <= 0xFFFF:
        token += struct.pack("!BH", 126, length)
    else:
        token += struct.pack("!BQ", 127, length)

    msg = token + msg_bytes
    conn.send(msg)
    return True

def recognize(record):
    global CONNECTION
    global DETECTOR
    global PREDICTOR
    global LIP_MODEL
    global COMMANDS
    global DRAGGING

    t1 = time.time()
    size = (96,48)  # 200*0.6*0.75
    lip = record[0][1]
    overall_h = int(lip[3] * 2.3* 6)  # 7.5*0.8
    overall_w = int(lip[2] * 1.8* 6)  #
    center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * 6
    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype('float32'))
    count = 0
    # cv2.namedWindow("window", cv2.WINDOW_NORMAL)  
    for entry in record:
        lip = entry[1]
        new_center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * 6
        if np.linalg.norm(new_center - center) < overall_h/2:
            center = new_center
        frame = entry[0]
        frame = cv2.resize(frame[center[1] - overall_h // 2:center[1] + overall_h // 2,
                                 center[0] - overall_w // 2:center[0] + overall_w // 2], size)
        # cv2.imshow("window",frame)
        # cv2.waitKey(33)
        buffer[count] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        count += 1
    t2 = time.time()
    print(f"frame processing time:{t2-t1}s")
    buffer = ((buffer - np.mean(buffer)) /
              np.std(buffer)).transpose(3, 0, 1, 2)
    buffer = torch.tensor(np.expand_dims(buffer, axis=0)).cuda()

    outputs = LIP_MODEL(buffer).cpu().detach().numpy()  
    t3 = time.time()
    print(f"inferenece time:{t3-t2}s")

    sorted_commands = sorted(list(zip(outputs[0], COMMANDS)))
    send_str = " ".join([s for _, s in sorted_commands])
    for s in sorted_commands:
        print(s)
    if DRAGGING:
        pyautogui.mouseUp()
        DRAGGING = False
    else:
        send_msg(CONNECTION,send_str.encode('utf-8')) 
        response = []
        data = get_data(CONNECTION.recv(8096))
        while data != "over":
            response.append(data)
            data = get_data(CONNECTION.recv(8096))
        if response:
            mixer.music.load(f'C:/Users/rkmtl/Documents/zxsu/sy_speech/{response[0]}.mp3')
            mixer.music.play()

            if response[0] == "drag":
                print(pyautogui.mouseDown())
                DRAGGING = True
            elif response[0] == "paste_here":
                pyautogui.hotkey("ctrl","v")
            elif response[0] == "copy_this":
                content = response[1]
                if content.startswith("https://"):
                    copy_image(content)
                else:
                    pyperclip.copy(content)
            print(response)
    t4 = time.time()
    print(f"nerwork communication & command execution time:{t4-t3}s")

def connect_web_socket(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', port))
    sock.listen(5)

    CONNECTION, address = sock.accept()
    data = CONNECTION.recv(1024)
    headers = get_headers(data)
    response_tpl = "HTTP/1.1 101 Switching Protocols\r\n" \
                   "Upgrade:websocket\r\n" \
                   "Connection:Upgrade\r\n" \
                   "Sec-WebSocket-Accept:%s\r\n" \
                   "WebSocket-Location:ws://%s%s\r\n\r\n"

    value = headers['Sec-WebSocket-Key'] + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
    ac = base64.b64encode(hashlib.sha1(value.encode('utf-8')).digest())
    response_str = response_tpl % (ac.decode('utf-8'), headers['Host'], headers['url'])
    CONNECTION.send(bytes(response_str, encoding='utf-8'))
    return CONNECTION

if __name__ == "__main__":
    COMMANDS = ['close_window',
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

    print("waiting for ws client...")
    CONNECTION = connect_web_socket(10130)
    print("reading face recognition model")
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("recognition model ready")
    print("reading lip model")
    LIP_MODEL = R2Plus1DClassifier(num_classes=18, layer_sizes=[3,3,2,2,2])
    state_dicts = torch.load(
        "demo40_web_model.pt_puremodel", map_location=torch.device("cuda:0"))
    LIP_MODEL.load_state_dict(state_dicts)
    LIP_MODEL.cuda()
    LIP_MODEL.eval()
    LIP_MODEL(torch.zeros(1,3,50,48,96,device="cuda:0"))
    print("lip model ready")

    print("camera preparing")
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap = VideoCapture(capture)
    print("camera ready")

    mixer.init()
    buffer = queue.Queue(maxsize=10)
    mouth_open = False
    record = []
    t1 = 0
    DRAGGING = False
    while True:
        frame = cap.read()
        image = cv2.resize(frame, (320, 180))
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
                if mo_angle > 0.1:
                    print("capturing speech")
                    mouth_open = True
                    record = list(buffer.queue)
                    buffer = queue.Queue(maxsize=10)
                    send_msg(CONNECTION,"mo".encode('utf-8')) 
            else:
                record.append([frame, lip])
                t1 = t1+1 if mo_angle < 0.1 else 0
            
            if t1 > 10 or len(record) == 90:
                print("speech finished")
                if len(record) > 30:
                    cap.recording = False
                    cap.q = queue.Queue(maxsize=100)
                    recognize(record)
                    cap.recording = True
                send_msg(CONNECTION,"mc".encode('utf-8')) 
                record = []
                t1 = 0
                mouth_open = False