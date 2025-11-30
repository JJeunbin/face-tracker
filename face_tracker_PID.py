from time import time
from ultralytics import YOLO
import cv2
import socket
import functions as ik
import numpy as np
import csv
import os

GREEN = (0, 255, 0)
RED = (0, 0, 255)

# ESP32의 SoftAP 기본 IP
ESP32_IP = "192.168.4.1"
ESP32_PORT = 5000

# UDP 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 카메라 시작
cap = cv2.VideoCapture(0)
# url = "http://192.168.4.4:8080/video"
# cap = cv2.VideoCapture(url)

# YOLO 모델 생성 / yolov8n-face pretrained ver
model = YOLO("yolov8n-face.pt")

# 카메라 연결 확인
print(cap.isOpened())

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# PID 중립각도
TH1_NEUTRAL = 90
TH2_NEUTRAL = 90
TH3_NEUTRAL = 110

prev_th1 = TH1_NEUTRAL
prev_th2 = TH2_NEUTRAL
prev_th3 = TH3_NEUTRAL
prev_th4 = 180 - TH3_NEUTRAL

# PID 설정 (초기값, 튜닝 필요)
Kp_x, Ki_x, Kd_x = 0.1, 0.01, 0.2   # th1 / del_x
Kp_y, Ki_y, Kd_y = 0.5, 0.01, 0.2   # th3 / del_y
Kp_d, Ki_d, Kd_d = 1.2, 0.02, 0.15  # th2 / 거리

# PID 상태
ix = iy = idist = 0.0
prev_del_x = prev_del_y = 0.0
prev_eD = 0.0
prev_time = time()
alpha = 0.45  # smoothing factor
max_deg_per_sec = 25.0  # 안전 속도 제한

# 거리 모델 (box_len → distance)
def boxlen_to_distance(box_len):
    return 0.1705 * box_len + 31.744  # 캘리브레이션 데이터 기반

# # CSV 기록 준비
# log_file = "pid_log.csv"
# csv_fields = ["time","del_x","del_y","box_len","D_curr","th1","th2","th3","th4"]
# if os.path.exists(log_file):
#     os.remove(log_file)
# csvfile = open(log_file, 'w', newline='')
# csvwriter = csv.DictWriter(csvfile, fieldnames=csv_fields)
# csvwriter.writeheader()


while True:
    start = time()              # 프레임 계산용 시작 시간 저장
    ret, frame = cap.read()
    if not ret:
        print("Cam Error")
        break

    frame = cv2.flip(frame, 1)      # 영상 좌우반전
    detection = model(frame, verbose=False)[0]     # 얼굴 감지 결과
    
    # 가장 가까운 얼굴 선택
    best_face = None
    best_size = -1

    for data in detection.boxes.xyxy:
        xmin, ymin, xmax, ymax = map(int, data)

        box_len = xmax - xmin     # 너가 기존에 사용하던 거리 판단 기준

        if box_len > best_size:
            best_size = box_len
            best_face = (xmin, ymin, xmax, ymax)

    if best_face is None:
        print("No face detected → using previous angles")
        message = f"th1:{prev_th1}, th2:{prev_th2}, th3:{prev_th3}, th4:{prev_th4}"
        sock.sendto(message.encode(), (ESP32_IP, ESP32_PORT))
        continue
    

    xmin, ymin, xmax, ymax = best_face
    xcenter, ycenter = (xmin+xmax)//2, (ymin+ymax)//2
    del_x = xcenter - 320
    del_y = 240 - ycenter
    box_len = xmax - xmin
    D_curr = boxlen_to_distance(box_len)
    D_target = 65.0         # 원하는 거리(cm)
    eD = D_target - D_curr
    
    # PID용 시간 계산
    now = time()
    dt = now - prev_time if now - prev_time > 1e-3 else 0.016
    prev_time = now

    # PID 계산
    # th1 : del_x
    ix += del_x*dt
    dx = (del_x - prev_del_x)/dt
    delta_th1 = Kp_x*del_x + Ki_x*ix + Kd_x*dx
    target_th1 = TH1_NEUTRAL + delta_th1
    prev_del_x = del_x

    # th3 : del_y
    iy += del_y*dt
    dy = (del_y - prev_del_y)/dt
    delta_th3 = Kp_y*del_y + Ki_y*iy + Kd_y*dy
    target_th3 = TH3_NEUTRAL + delta_th3
    prev_del_y = del_y

    # th2 : distance
    idist += eD*dt
    idist = np.clip(idist,-50,50)
    ded = (eD - prev_eD)/dt
    delta_th2 = Kp_d*eD + Ki_d*idist + Kd_d*ded
    target_th2 = TH2_NEUTRAL + delta_th2
    prev_eD = eD

    # rate limit
    max_step = max_deg_per_sec * dt
    target_th1 = np.clip(target_th1, prev_th1 - max_step, prev_th1 + max_step)
    target_th2 = np.clip(target_th2, prev_th2 - max_step, prev_th2 + max_step)
    target_th3 = np.clip(target_th3, prev_th3 - max_step, prev_th3 + max_step)

    # smoothing + limits
    th1 = int(np.clip(alpha*target_th1 + (1-alpha)*prev_th1, 0, 180) + 0.5)
    th2 = int(np.clip(alpha*target_th2 + (1-alpha)*prev_th2, 0, 180) + 0.5)
    th3 = int(np.clip(alpha*target_th3 + (1-alpha)*prev_th3, 0, 180) + 0.5)

    # th4: 화면 수평 유지
    th4 = 180 - th3
    if th3 >= 125: th3 = 125
    
    # th1 = 90
    # th2 = 90
    # th3 = 90
    # th4 = 90
    
    # ESP32로 좌표 전송
    message = f"th1:{th1}, th2:{th2}, th3:{th3}, th4:{th4}"
    sock.sendto(message.encode(), (ESP32_IP, ESP32_PORT))
    print("Sent:", message)

    # 이전 각도 갱신
    prev_th1, prev_th2, prev_th3, prev_th4 = th1, th2, th3, th4
    

    # 화면 표시
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
    cv2.circle(frame, (xcenter, ycenter), 2, GREEN, 3)
    cv2.putText(frame, f"({xcenter}, {ycenter})", (xcenter+10, ycenter+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    cv2.circle(frame, (320, 240), 2, RED, 3)
    cv2.putText(frame, f"({320}, {240})", (320+10, 240+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
    fps = f"FPS:{1 / (time()-start):.2f}"
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
    cv2.imshow("vedio", frame)
    
    # # --- CSV 기록 ---
    # csvwriter.writerow({
    #     "time": now,
    #     "del_x": del_x,
    #     "del_y": del_y,
    #     "box_len": box_len,
    #     "D_curr": D_curr,
    #     "th1": th1,
    #     "th2": th2,
    #     "th3": th3,
    #     "th4": th4
    # })

    if cv2.waitKey(1) == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()