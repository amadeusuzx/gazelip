import cv2
import threading
import queue
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_EXPOSURE,-6)
        print(self.cap.set(cv2.CAP_PROP_FPS, 10))
        self.recording = True
        self.q = queue.Queue(maxsize=100)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()   
        
    def _reader(self):
        size = (1280,720)
        # fourcc = cv2.VideoWriter_fourcc(*'I420')
        # fps = 30
        # save_name = "./test.avi"
        # video_writer = cv2.VideoWriter(save_name, fourcc, fps, size)
        while True:

            if self.recording:
                ret, frame = self.cap.read()
                if not ret:
                    print("camera faliure")
                # video_writer.write(frame)
                else:
                    self.q.put_nowait(frame)
            else:
                time.sleep(0.1)

    def read(self):
        return self.q.get()
import cv2
import sys

camera_id = 0
delay = 1
window_name = 'frame'
cap = VideoCapture(camera_id)
tm = cv2.TickMeter()
tm.start()

count = 0
max_count = 10
fps = 0

while True:
    frame = cap.q.get()

    if count == max_count:
        tm.stop()
        fps = max_count / tm.getTimeSec()
        tm.reset()
        tm.start()
        count = 0

    cv2.putText(frame, 'FPS: {:.2f}'.format(fps),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv2.imshow(window_name, frame)
    count += 1

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)