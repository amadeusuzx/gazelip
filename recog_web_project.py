import os
import time

import numpy as np
import cv2
import dlib
from imutils import face_utils
from network import R2Plus1DClassifier
import torch

import threading
import queue

import onnxruntime
import pyautogui
import socketserver

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
        size = (1280, 720)
        fourcc = cv2.VideoWriter_fourcc(*'I420')
        fps = 30
        save_name = "./test.avi"
        video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
        while True:

            if self.recording:
                ret, frame = self.cap.read()
                if not ret:
                    print("camera faliure")
                    break
                video_writer.write(frame)
                self.q.put_nowait(frame)
            else:
                time.sleep(0.1)

    def read(self):
        return self.q.get()


def recognize(record):
    size = (96,48)  # 200*0.6*0.75
    lip = record[0][1]
    overall_h = int(lip[3] * 2.3* 5*0.8)  # 6*0.75
    overall_w = int(lip[2] * 1.8* 5*0.8)  # 6*0.75
    center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * 4
    buffer = np.empty((len(record), size[1], size[0], 3), np.dtype('float32'))
    count = 0
    # cv2.namedWindow("window", cv2.WINDOW_NORMAL)  
    for entry in record:
        lip = entry[1]
        new_center = np.array((lip[0] + lip[2]//2, lip[1] + lip[3]//2)) * 4
        if np.linalg.norm(new_center - center) < overall_h/2:
            center = new_center
        frame = entry[0]
        frame = cv2.resize(frame[center[1] - overall_h // 2:center[1] + overall_h // 2,
                                 center[0] - overall_w // 2:center[0] + overall_w // 2], size)
        # cv2.imshow("window",frame)
        # cv2.waitKey(33)
        buffer[count] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        count += 1

    buffer = ((buffer - np.mean(buffer)) /
              np.std(buffer)).transpose(3, 0, 1, 2)
    buffer = torch.tensor(np.expand_dims(buffer, axis=0)).cuda()

    outputs = lip_model(buffer).cpu().detach().numpy()  
    commands = ['close_window',
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
    pred_index = outputs[0].argmax()
    sorted_commands = sorted(list(zip(outputs[0], commands)))
    send_str = " ".join([s for _, s in sorted_commands])
    for s in sorted_commands:
        print(s)
    send_msg(conn,send_str.encode('utf-8')) 
    data = get_data(conn.recv(8096))
    if data:
        pass

        
def get_data(info):
    payload_len = info[1] & 127
    if payload_len == 126:
        extend_payload_len = info[2:4]
        mask = info[4:8]
        decoded = info[8:]
    elif payload_len == 127:
        extend_payload_len = info[2:10]
        mask = info[10:14]
        decoded = info[14:]
    else:
        extend_payload_len = None
        mask = info[2:6]
        decoded = info[6:]

    bytes_list = bytearray() 
    for i in range(len(decoded)):
        chunk = decoded[i] ^ mask[i % 4]
        bytes_list.append(chunk)
    body = str(bytes_list, encoding='utf-8')
    return body

def get_headers(data):

    header_dict = {}
    data = str(data, encoding='utf-8')

    header, body = data.split('\r\n\r\n', 1)
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


if __name__ == "__main__":

    import socket
    import base64
    import hashlib

    global conn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 10130))
    sock.listen(5)

    conn, address = sock.accept()
    data = conn.recv(1024)
    headers = get_headers(data)
    response_tpl = "HTTP/1.1 101 Switching Protocols\r\n" \
                   "Upgrade:websocket\r\n" \
                   "Connection:Upgrade\r\n" \
                   "Sec-WebSocket-Accept:%s\r\n" \
                   "WebSocket-Location:ws://%s%s\r\n\r\n"

    value = headers['Sec-WebSocket-Key'] + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
    ac = base64.b64encode(hashlib.sha1(value.encode('utf-8')).digest())
    response_str = response_tpl % (ac.decode('utf-8'), headers['Host'], headers['url'])
    conn.send(bytes(response_str, encoding='utf-8'))

    global detector
    global predictor

    print("reading face recognition model")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("recognition model ready")

    from network import R2Plus1DClassifier
    import torch

    global lip_model
    print("reading lip model")
    lip_model = R2Plus1DClassifier(num_classes=18, layer_sizes=[3,3,2,2,2])
    state_dicts = torch.load(
        "demo35.pt", map_location=torch.device("cuda:0"))
    lip_model.load_state_dict(state_dicts)
    lip_model.cuda()
    lip_model.eval()
    lip_model(torch.zeros(1,3,50,48,96,device="cuda:0"))
    print("lip model ready")

    print("camera preparing")
    cap = VideoCapture(0)
    print("camera ready")

    buffer = queue.Queue(maxsize=15)
    mo = False
    record = []

    j = 0
    t1 = 0
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
                lip = cv2.boundingRect(shape[48:68])
                angle = np.linalg.norm(
                    shape[62] - shape[66]) / np.linalg.norm(shape[60] - shape[64])
                buffer.put_nowait([frame, lip])
                if angle > 0.1:
                    if not mo:
                        print("capturing speech")
                        mo = True
                    t1 = 0
                if mo and angle < 0.1:
                    t1 += 1
                if t1 > 15 or len(record) == 90:
                    mo = False
                    print("speech finished")
                    if len(record) > 30:
                        cap.recording = False
                        cap.q = queue.Queue(maxsize=100)
                        recognize(record)
                        cap.recording = True
                    record = []
                    t1 = 0
