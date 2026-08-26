import cv2
import numpy as np
import mediapipe as mp
import math
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def distance(p1,p2):
    return math.hypot(p2[0]-p1[0],p2[1]-p1[1])

# --------------- OpenCV Kalman Filter 설정 ---------------
# 상태 벡터: [x, y, vx, vy]
# 측정 벡터: [x, y]
kalman = cv2.KalmanFilter(4, 2)

# Transition Matrix (상태 전이 행렬)
kalman.transitionMatrix = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

# Measurement Matrix (측정 행렬)
# 측정값으로부터 x, y만 관측
kalman.measurementMatrix = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ],
    dtype=np.float32,
)

# Process Noise Covariance Matrix (프로세스 노이즈 공분산)
# 값이 클수록 모델 예측보다 측정값 변화에 더 유연하게 반응
kalman.processNoiseCov = np.array(
    [
        [1e-2, 0, 0, 0],
        [0, 1e-2, 0, 0],
        [0, 0, 5e-2, 0],
        [0, 0, 0, 5e-2],
    ],
    dtype=np.float32,
)

# Measurement Noise Covariance Matrix (측정 노이즈 공분산)
# 값이 클수록 측정값을 덜 신뢰하고 예측값을 더 신뢰
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

# Posteriori Error Covariance Matrix (초기 추정 오차 공분산)
kalman.errorCovPost = np.eye(4, dtype=np.float32)

# -------------------------------------------------------

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

# Morphological operation용 타원형 kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 객체 최초 검출 여부 확인용 boolean
found = False

# 객체 좌표 및 반지름 변수 정의
x_bel, y_bel = 0, 0
radius = 0


base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보
finger_tips = (4, 8, 12, 16, 20)                                                                # 손가락 끝 landmark의 index


pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
center=(0,0)
point1=(0,0)
radius = 0
while True:
    start = time.time()
    ret, frame = cap.read()

    if not ret:
            break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame = cv2.flip(frame, 0)

    # 가우시안 블러
    blr = cv2.GaussianBlur(frame, (11, 11), 0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # LAB 색공간 변환
    lab = cv2.cvtColor(blr, cv2.COLOR_BGR2LAB)

    # LAB color segmentation
    mask = cv2.inRange(lab, green_lower, green_upper)

    # Opening 2회, Dilation 2회
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 프레임 내 손 탐지
    result = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    
    # 화면 좌측 상단에 텍스트 생성 (손 개수, 왼손/오른손/양손 여부)
    labels = ["Left" if h[0].category_name == "Right" else "Right" for h in result.handedness]
    cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, " / ".join(labels), (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    # 탐지 결과의 각 손마다 선과 점 그리기
    for hand in result.hand_landmarks:
        h, w = frame.shape[:2]  # 프레임 높이와 너비
        points = [(int(p.x * w), int(p.y * h)) for p in hand]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표

        # landmark를 연결하는 선 (skeleton) 그리기
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], (0,255,0), 2)
    
        # 각 관절 (landmark)에 점 그리기 (손가락 끝은 빨간 점, 그 외에는 파란 점)
        for i, point in enumerate(points):
            if i in finger_tips:
                color=(0,0,255)
                point1=points[i]
            else:
                color = (255,0,0)
            cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)

    # Contour detection
    contour_lst, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour_lst) > 0:
        # 가장 큰 contour 선택
        contour = max(contour_lst, key=cv2.contourArea)
        # 최소 외접원 반지름
        _, radius = cv2.minEnclosingCircle(contour)
        # 무게중심
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (0, 0)

        # 검출된 객체에 파란 원 overlay
        cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

        # Measurement (측정값)
        z = np.array([[center[0]], [center[1]]],dtype=np.float32)
        radius=radius

        # 객체 최초 검출 (최초 a priori state 저장)
        if not found:
            kalman.statePre = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)
            found = True

    # 최초 검출 이후 Kalman Filter 적용
    # 측정값 사용 (visible) -> Prediction & Update
    if found and (len(contour_lst) > 0):
        predicted_state = kalman.predict()   # prediction
        corrected_state = kalman.correct(z)  # update
        x_bel = corrected_state[0, 0]
        y_bel = corrected_state[1, 0]

    # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contour_lst) <= 0):
        predicted_state = kalman.predict()   # prediction (update X)
        x_bel = predicted_state[0, 0]
        y_bel = predicted_state[1, 0]

    # 예측한 객체 위치에 노란 원 overlay
    cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (0, 255, 255), 2)
    cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (0, 255, 255), -1)

    if distance(center,point1) < radius+10.0:
        cv2.putText(frame, f"Grab Ball", (20,105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Object Detection", frame)
    # cv2.imshow("LAB Mask", mask)

    # while loop rate (FPS) 설정
    time.sleep(max(1. / 25 - (time.time() - start), 0))


hand_detector.close()
cap.release()
cv2.destroyAllWindows()