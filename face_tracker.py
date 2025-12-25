import cv2
import threading
from time import time
from ultralytics import YOLO
import socket

GREEN = (0, 255, 0)
RED = (0, 0, 255)

GAIN_X = 0.011
GAIN_Y = 0.01
GAIN_Z = 0.05

DEADZONE_X = 20
DEADZONE_Y = 15
DEADZONE_Z = 10

# -----------------------------
# UDP 설정
# -----------------------------
ESP32_IP = "192.168.4.1"
ESP32_PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------------
# YOLO 모델 불러오기
# -----------------------------
model = YOLO("yolov8n-face.pt")

# -----------------------------
# 최신 프레임만 유지하는 클래스
# -----------------------------
class FrameGetter:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                ret, frame = self.cap.read()

                # ret이 False거나 frame이 None이면 그냥 스킵
                if not ret or frame is None:
                    continue

                with self.lock:
                    self.frame = frame

            except Exception as e:
                print("Frame read error:", e)
                continue

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# -----------------------------
# 프레임 수신 스레드 시작
# -----------------------------
url = "http://192.168.4.2:8080/video"
frame_getter = FrameGetter(url).start()

print("Camera opened:", frame_getter.cap.isOpened())

# -----------------------------
# 서보 초기값
# -----------------------------
sock.sendto("REQ_ANGLE".encode(), (ESP32_IP, ESP32_PORT))
data, _ = sock.recvfrom(1024)
th1, th2, th3 = map(int, data.decode().split(','))
print("Initial angles from ESP32:", th1, th2, th3)

# -----------------------------
# 메인 루프
# -----------------------------
while True:
    start = time()

    frame = frame_getter.read()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)

    # YOLO 추론
    detection = model(frame, verbose=False)[0]

    best_face = None
    best_size = -1

    for data in detection.boxes.xyxy:
        xmin, ymin, xmax, ymax = map(int, data)
        sz = xmax - xmin
        if sz <= 50:
            continue
        if sz > best_size:
            best_size = sz
            best_face = (xmin, ymin, xmax, ymax)
    
    if best_face is None:
        cv2.putText(frame, "No Face Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

    else:
        xmin, ymin, xmax, ymax = best_face
        xcenter, ycenter = (xmin+xmax)//2, (ymin+ymax)//2

        # 오차 계산
        del_x = xcenter - 320
        del_y = 240 - ycenter
        box_len = xmax - xmin
        target_size = 140
        size_error = box_len - target_size

        # 좌우
        if abs(del_x) < DEADZONE_X: del_x = 0
        elif abs(del_x) < 30: GAIN_X = 0.01
        elif abs(del_x) > 200: GAIN_X = 0.005
        th1 += del_x * GAIN_X

        # 상하
        if abs(del_y) < DEADZONE_Y: del_y = 0
        # elif abs(del_y) > 
        th3 += del_y * GAIN_Y
        th3 = max(50, min(130, th3))

        # 거리 조절
        if abs(size_error) < DEADZONE_Z: size_error = 0
        elif abs(size_error) < 10: GAIN_Z = 0.05
        th2 -= size_error * GAIN_Z
        th2 = max(30, min(135, th2))

        # 서보 범위
        th1 = max(0, min(180, th1))
        th2 = max(0, min(180, th2))
        th3 = max(0, min(180, th3))

        # ESP32로 전송
        message = f"th1:{int(th1)}, th2:{int(th2)}, th3:{int(th3)}"
        sock.sendto(message.encode(), (ESP32_IP, ESP32_PORT))
        print(message)

        # 화면 표시용
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.circle(frame, (xcenter, ycenter), 2, GREEN, 3)
        cv2.putText(frame, f"({xcenter}, {ycenter})", (xcenter+10, ycenter+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        cv2.putText(frame, f"{box_len}", (xcenter-10, ymax+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        cv2.circle(frame, (320, 240), 2, RED, 3)
        cv2.putText(frame, f"({320}, {240})", (320+10, 240+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    # FPS 표시
    fps = f"FPS: {1 / (time() - start):.2f}"
    cv2.putText(frame, fps, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

frame_getter.stop()
cv2.destroyAllWindows()