import subprocess
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('127.0.0.1', 50007))
    s.listen(1)
    gaze = None
    while True:
        conn, _ = s.accept()
        print("console connected")
        conn.settimeout(3)
        with conn:
            while True:
                try:
                    data = conn.recv(1024)
                    if data == b"g":
                        gaze = subprocess.Popen("GazeCursorClick.exe")
                        print("gaze started")
                    elif data == b"s":
                        if gaze:
                            gaze.kill()
                            print("gaze stoped")
                except:
                    pass
                

    
