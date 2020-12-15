
import socket
import msvcrt

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 50007        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("connected")
    while True:
        data = msvcrt.getch()
        if data == b'c':
            break
        else:
            s.sendall(data)
