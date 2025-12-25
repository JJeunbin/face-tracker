from time import time
from ultralytics import YOLO
import cv2
import socket
import os
import csv
import numpy as np
import functions as fnc


GREEN = (0, 255, 0)
RED = (0, 0, 255)

# ESP32
ESP32_IP = "192.168.4.1"
ESP32_PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# YOLO 모델 생성 / yolov8n-face pretrained ver
model = YOLO("yolov8n-face.pt")

# 카메라
# cap = cv2.VideoCapture(0)
# url = "http://10.221.151.220:8080/video"
url = "http://192.168.4.4:8080/video"
cap = cv2.VideoCapture(url)
print(cap.isOpened())
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)      # 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 로봇 링크 및 최대 각속도 제한
L2, L3, H = 17, 16, 5
MAX_ANGULAR_VELOCITY = 40.0

# 거리 모델 (box_len → distance(cm))
def boxlen_to_distance(box_len):
    return -0.194*box_len + 82.79

# PID 초기값
alpha = 0.45
dt_prev = time()
prev_del_x = prev_del_y = prev_del_z = 0
prev_err_x = prev_err_y = prev_err_z = 0.0

# PD 계수 (좌표 제어용)
Kp_x, Kd_x = 0.05, 0.02
Kp_y, Kd_y = 0.05, 0.02
Kp_z, Kd_z = 0.03, 0.02

# 초기 좌표
Px = 15.0
Py = 0.0
Pz = 25.0

prev_th1 = prev_th2 = prev_th3 = prev_th4 = 90

# CSV 기록 준비
log_file = "pid_log.csv"
csv_fields = ["time","err_x","err_y","err_z","Px","Py","Pz","th1","th2","th3"]
if os.path.exists(log_file):
    os.remove(log_file)
csvfile = open(log_file, 'w', newline='')
csvwriter = csv.DictWriter(csvfile, fieldnames=csv_fields)
csvwriter.writeheader()


while True:
    start = time()              # 프레임 계산용 시작 시간 저장
    ret, frame = cap.read()
    if not ret:
        print("Cam Error")
        break

    frame = cv2.flip(frame, 1)                      # 영상 좌우반전
    detection = model(frame, verbose=False)[0]      # 얼굴 감지 결과
    
    # 가장 가까운 얼굴 선택
    best_face = None
    best_size = -1
    for data in detection.boxes.xyxy:
        xmin, ymin, xmax, ymax = map(int, data)
        box_len = xmax - xmin
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
    box_len = xmax - xmin

    # 화면 표시
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
    cv2.circle(frame, (xcenter, ycenter), 2, GREEN, 3)
    cv2.putText(frame, f"({xcenter}, {ycenter})", (xcenter+10, ycenter+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    cv2.putText(frame, f"{box_len}", (xcenter-10, ymax+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    cv2.circle(frame, (320, 240), 2, RED, 3)
    cv2.putText(frame, f"({320}, {240})", (320+10, 240+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    # Px = 15 - (box_len - 130) / 5
    # Py = 2 * del_x / 320
    # Pz = 25  + 5 * del_y / 240
    # if Pz < 33.5:
    #     Pz = 25 + 5 * del_y / 240
    # else:
    #     Pz = 33.5

    # PID 시간 계산
    now = time()
    dt = max(now - dt_prev, 1e-3)
    dt_prev = now

    # --- 좌표 PID 제어 ---
    del_x = xcenter - 320
    del_y = 240 - ycenter
    del_z = 60 - boxlen_to_distance(box_len)
    del_x, del_y, del_z = fnc.filter_del(del_x, del_y, del_z, prev_del_x, prev_del_y, prev_del_z)

    err_x = del_z / 10
    err_y = 0.01 * del_x * Px
    err_z = 0.01 * del_y * Px
    err_x = 0
    # err_y = 0
    # err_z = 0

    dx = (err_x - prev_err_x)/dt
    dy = (err_y - prev_err_y)/dt
    dz = (err_z - prev_err_z)/dt

    # 좌표 업데이트
    Px += Kp_x*err_x + Kd_x*dx
    Py += Kp_y*err_y + Kd_y*dy
    Pz += Kp_z*err_z + Kd_z*dz

    prev_err_x = err_x
    prev_err_y = err_y
    prev_err_z = err_z

    Px, Py, Pz = fnc.limit_workspace(Px, Py, Pz, L2, L3, H)
    print("Px:", Px, "Py:", Py, "Pz:", Pz)
    
    try:
        th1, th2, th3 = fnc.inverse_kinematics(Px, Py, Pz, L2, L3, H)
        if th3 >= 130: th3 = 130
        # th4 = 180 - th3        

        # 최대 허용 각속도 반영
        max_delta_theta = MAX_ANGULAR_VELOCITY * dt

        delta_th1 = th1 - prev_th1
        delta_th2 = th2 - prev_th2
        delta_th3 = th3 - prev_th3
        delta_th4 = th4 - prev_th4

        th1 = int(prev_th1 + np.clip(delta_th1, -max_delta_theta, max_delta_theta))
        th2 = int(prev_th2 + np.clip(delta_th2, -max_delta_theta, max_delta_theta))
        th3 = int(prev_th3 + np.clip(delta_th3, -max_delta_theta, max_delta_theta))
        th4 = int(prev_th4 + np.clip(delta_th4, -max_delta_theta, max_delta_theta))

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
    # print("Sent:", message)

    # FPS 표시
    end = time()
    fps = f"FPS: {1/(end-start):.2f}"
    cv2.putText(frame, fps, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
    cv2.imshow("video", frame)

    # --- CSV 기록 ---
    csvwriter.writerow({
        "time": end-start,
        "err_x": err_x,
        "err_y": err_y,
        "err_z": err_z,
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "th1": th1,
        "th2": th2,
        "th3": th3,
    })

    if cv2.waitKey(1) == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()