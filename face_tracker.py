from time import time
from ultralytics import YOLO
import cv2
import socket
import inverse_kinematics as ik
import numpy as np

GREEN = (0, 255, 0)
RED = (0, 0, 255)

# ESP32의 SoftAP 기본 IP
ESP32_IP = "192.168.4.1"
ESP32_PORT = 5000

# UDP 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 카메라 시작
# cap = cv2.VideoCapture(0)
url = "http://192.168.4.4:8080/video"
cap = cv2.VideoCapture(url)

# YOLO 모델 생성 / yolov8n-face pretrained ver
model = YOLO("yolov8n-face.pt")

# 카메라 연결 확인
print(cap.isOpened())

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_th1 = 90
prev_th2 = 90
prev_th3 = 90
prev_th4 = 90

while True:
    start = time()              # 프레임 계산용 시작 시간 저장
    ret, frame = cap.read()
    if not ret:
        print("Cam Error")
        break

    frame = cv2.flip(frame, 1)      # 영상 좌우반전
    detection = model(frame, verbose=False)[0]     # 얼굴 감지 결과
    
    # ----- 여러 얼굴 중 가장 가까운 얼굴 선택 -----
    best_face = None
    best_size = -1

    for data in detection.boxes.xyxy:
        xmin, ymin, xmax, ymax = map(int, data)

        box_len = xmax - xmin     # 너가 기존에 사용하던 거리 판단 기준

        if box_len > best_size:
            best_size = box_len
            best_face = (xmin, ymin, xmax, ymax)

    # ----- 얼굴이 하나도 안 잡혔을 때 -----
    if best_face is None:
        print("No face detected → using previous angles")
        message = f"th1:{prev_th1}, th2:{prev_th2}, th3:{prev_th3}, th4:{prev_th4}"
        sock.sendto(message.encode(), (ESP32_IP, ESP32_PORT))
        continue
    
    xmin, ymin, xmax, ymax = best_face
    xcenter, ycenter = (xmin+xmax)//2, (ymin+ymax)//2
    # print("face center position:", xcenter, ",", ycenter)
    
    # 박스 그리기, 중심점 찍기
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
    cv2.circle(frame, (xcenter, ycenter), 2, GREEN, 3)
    cv2.putText(frame, f"({xcenter}, {ycenter})", (xcenter+10, ycenter+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    # 화면 중심점 찍기
    cv2.circle(frame, (320, 240), 2, RED, 3)
    cv2.putText(frame, f"({320}, {240})", (320+10, 240+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    del_x = xcenter - 320
    del_y = 240 - ycenter
    box_len = xmax - xmin

    L2 = 17
    L3 = 16
    H = 5

    Px = 15 - (box_len - 130) / 5
    Py = 2 * del_x / 320
    Pz = 30 + 5 * del_y / 240
    if Pz < 33.5:
        Pz = 30 + 10 * del_y / 240
    else:
        Pz = 33.5

    # print("Px:", Px, "Py:", Py, "Pz:", Pz)


     # ----- inverse_kinematics 실패 대비 -----
    try:
        th1, th2, th3, = ik.inverse_kinematics(Px, Py, Pz, L2, L3, H)
        th4 = 180 - th3
        th4 = 20 + (th4 * 140 / 180) 

        # 계산 성공 → 이전 각도 갱신
        prev_th1, prev_th2, prev_th3, prev_th4 = th1, th2, th3, th4

    except Exception as e:
        print("IK failed → using previous angles:", e)
        th1, th2, th3, th4 = prev_th1, prev_th2, prev_th3, prev_th4
    
    # th1 = 90
    # th2 = 90
    # th3 = 90
    # th4 = 90

    # ESP32로 좌표 전송
    message = f"th1:{th1}, th2:{th2}, th3:{th3}, th4:{th4}"
    sock.sendto(message.encode(), (ESP32_IP, ESP32_PORT))
    #print("Sent:", message)

    end = time()            # 프레임 계산용 종료 시간 저장
    total = end - start                                                         # 총 처리 시간 (초단위)
    # print(f"Time to process 1 frame: {total:.2f} seconds")
    fps = f"FPS: {1 / total:.2f}"                                               # 프레임 계산
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)    # 프레임 화면 출력
    
    # 윈도우 창 띄우기
    cv2.namedWindow("vedio", cv2.WINDOW_NORMAL)
    cv2.imshow("vedio", frame)
    
    if cv2.waitKey(1) == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()