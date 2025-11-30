from time import time
from ultralytics import YOLO
import cv2
import socket
import functions as ik
import numpy as np

GREEN = (0, 255, 0)
RED = (0, 0, 255)

# ESP32의 SoftAP 기본 IP
ESP32_IP = "192.168.4.1"
ESP32_PORT = 5000

# UDP 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 카메라 시작
url = "http://192.168.4.4:8080/video"
cap = cv2.VideoCapture(url)

# YOLO 모델 생성 / yolov8n-face pretrained ver
model = YOLO("yolov8n-face.pt")

print(cap.isOpened())

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==============================
#   이전 각도 저장 변수
# ==============================
prev_th1 = 90
prev_th2 = 90
prev_th3 = 90
prev_th4 = 90

# ==============================
#   필터용 초기 좌표 값
# ==============================
smooth_Px = 15
smooth_Py = 0
smooth_Pz = 20

alpha = 0.12   # 필터 강도 (0.05~0.2 사이 추천)
deadzone_px = 6     # X Deadzone (픽셀)
deadzone_py = 6     # Y Deadzone (픽셀)

while True:
    start = time()
    ret, frame = cap.read()
    if not ret:
        print("Cam Error")
        break

    frame = cv2.flip(frame, 1)
    detection = model(frame, verbose=False)[0]

    # ----- 여러 얼굴 중 가장 가까운 얼굴 선택 -----
    best_face = None
    best_size = -1

    for data in detection.boxes.xyxy:
        xmin, ymin, xmax, ymax = map(int, data)
        box_len = xmax - xmin

        if box_len > best_size:
            best_size = box_len
            best_face = (xmin, ymin, xmax, ymax)

    # ----- 얼굴 없음 → 이전 각도로 유지 -----
    if best_face is None:
        print("No face detected → using previous angles")
        message = f"th1:{prev_th1}, th2:{prev_th2}, th3:{prev_th3}, th4:{prev_th4}"
        sock.sendto(message.encode(), (ESP32_IP, ESP32_PORT))
        continue

    xmin, ymin, xmax, ymax = best_face
    xcenter, ycenter = (xmin+xmax)//2, (ymin+ymax)//2

    # Draw
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
    cv2.circle(frame, (xcenter, ycenter), 2, GREEN, 3)
    cv2.putText(frame, f"({xcenter}, {ycenter})", (xcenter+10, ycenter+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    cv2.circle(frame, (320, 240), 2, RED, 3)

    # ==============================
    #   ΔX, ΔY 계산 + Deadzone 적용
    # ==============================
    del_x = xcenter - 320
    del_y = 240 - ycenter
    box_len = xmax - xmin

    # Deadzone
    if abs(del_x) < deadzone_px:
        del_x = 0
    if abs(del_y) < deadzone_py:
        del_y = 0

    # ==============================
    #   Raw Px, Py, Pz 계산
    # ==============================
    # Px_raw = 15 - (box_len - 130) / 5
    # Py_raw = 2 * del_x / 320
    # Pz_raw = 30 + 5 * del_y / 240
    Px_raw = 15
    Py_raw = 0
    Pz_raw = 20
    Pz_raw = min(Pz_raw, 33.5)

    # ==============================
    #   1차 LPF 적용 (Exponential Smoothing)
    # ==============================
    smooth_Px = smooth_Px * (1 - alpha) + Px_raw * alpha
    smooth_Py = smooth_Py * (1 - alpha) + Py_raw * alpha
    smooth_Pz = smooth_Pz * (1 - alpha) + Pz_raw * alpha

    # ==============================
    #   Inverse Kinematics
    # ==============================
    try:
        th1, th2, th3 = ik.inverse_kinematics(smooth_Px, smooth_Py, smooth_Pz, 17, 16, 5)
        th4 = 180 - th3
        # th4 = 20 + (th4 * 140 / 180)

        prev_th1, prev_th2, prev_th3, prev_th4 = th1, th2, th3, th4

    except Exception as e:
        print("IK failed → using previous angles:", e)
        th1, th2, th3, th4 = prev_th1, prev_th2, prev_th3, prev_th4

    # ==============================
    #   Send to ESP32
    # ==============================
    message = f"th1:{th1}, th2:{th2}, th3:{th3}, th4:{th4}"
    sock.sendto(message.encode(), (ESP32_IP, ESP32_PORT))

    # FPS 표시
    fps = f"FPS: {1 / (time() - start):.2f}"
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    cv2.imshow("vedio", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()