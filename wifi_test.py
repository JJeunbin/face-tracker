import cv2
import socket

# ESP32의 SoftAP 기본 IP
ESP32_IP = "192.168.4.1"
ESP32_PORT = 5000

# UDP 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 카메라 시작
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2

        # 얼굴 위치 표시
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # ESP32로 좌표 전송
        message = f"x:{cx},y:{cy}"
        sock.sendto(message.encode(), (ESP32_IP, ESP32_PORT))

        print("Sent:", message)
        break  # 한 번에 한 얼굴만 처리

    cv2.imshow('Face Tracking', frame)
    if cv2.waitKey(1) == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()