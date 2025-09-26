from time import time, sleep    # 코드 지연 및 프레임 계산 / 시간 관련 기능들
import cv2                      # opencv 패키지
from ultralytics import YOLO    # YOLO 패키지

GREEN = (0, 255, 0)
RED = (0, 0, 255)

# YOLO 모델 생성 / yolov8n-face pretrained ver
model = YOLO("yolov8n-face.pt")


# videoCapture 시작 / 카메라 연결
cap = cv2.VideoCapture(0)
# url = "http://10.221.52.72:8080/video"
cap = cv2.VideoCapture(url)

# 카메라 연결 확인
print(cap.isOpened())

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    # 프레임 계산용 시작 시간 저장
    start = time()
    
    # ret = true 영상 정상적으로 읽어옴 frame = 프레임 넘파이 행렬 데이터
    ret, frame = cap.read()

    # ret = false일 경우 에러 처리
    if not ret:
        print("Cam Error")
        break
    
    frame = cv2.flip(frame, 1)              # 영상 좌우반전
    detection = model(frame)[0]             # 얼굴 감지 결과 받아오기
    
    for data in (detection.boxes.xyxy):     # 박스 데이터 리스트 변환
                                            # data : [xmin, ymin, xmax, ymax]
        # 박스 좌표 입력
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        xcenter, ycenter = (xmin+xmax)//2, (ymin+ymax)//2
        print("face center position:", xcenter, ",", ycenter)
        
        # 박스 그리기, 중심점 찍기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.line(frame, (xcenter, ycenter), (xcenter, ycenter), GREEN, 5)
        

    # 프레임 계산용 종료 시간 저장
    end = time()

    # 총 처리 시간 (초단위)
    total = end - start
    print(f"Time to process 1 frame: {total:.2f} seconds")

    # 프레임 계산
    fps = f"FPS: {1 / total:.2f}"

    # 프레임 화면 출력
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)    

    # 윈도우 창 띄우기
    cv2.namedWindow("vedio", cv2.WINDOW_NORMAL)
    cv2.imshow("vedio", frame)
    
    # q 누르면 무한 루프 종료
    if cv2.waitKey(1) == ord("q"):
        break


cap.release()               # 카메라 연결 해제
cv2.destroyAllWindows()     # 프로그램 종료