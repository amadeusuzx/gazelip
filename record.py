import cv2
import sys
import dlib
from PIL import Image
from imutils import face_utils
import numpy as np
import time
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

camera_id = 1
delay = 1
window_name = 'frame'

cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

size = (1920,1080)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30
save_name = "./test.avi"
video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
if not cap.isOpened():
    sys.exit()
while cap.isOpened():
    ret, frame = cap.read()
    video_writer.write(frame)