import socket
import time

GREEN = (0, 255, 0)
RED = (0, 0, 255)

# ESP32의 SoftAP 기본 IP
ESP32_IP = "192.168.4.1"
ESP32_PORT = 5000

# UDP 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    th1 = 90
    th2 = 90
    th3 = 90
    th4 = 90

    
    message = f"th1:{th1}, th2:{th2}, th3:{th3}, th4:{th4}"
    sock.sendto(message.encode(), (ESP32_IP, ESP32_PORT))
    print("Sent:", message)
    time.sleep(0.1)