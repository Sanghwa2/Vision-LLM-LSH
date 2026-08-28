
# [ANSWER] 카메라 프레임 전체 화면의 YOLO 탐지 결과를 Dictionary → JSON → Natural Language → Gemma 순서로 전달

import cv2
import json
import numpy as np
import mediapipe as mp
import tts

from ultralytics import YOLO
from llama_cpp import Llama
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS                                  # 각 landmark를 잇는 연결선 정보
finger_tips = (4, 8, 12, 16, 20)                                                                # 손가락 끝 landmark의 index

def get_nearest_pointed_object(fingertip,prev_joint,boxes,result,max_angle=60):
    fingertip = np.array(fingertip,dtype=float)
    prev_joint = np.array(prev_joint,dtype=float)

    finger_dir = fingertip - prev_joint
    norm = np.linalg.norm(finger_dir)

    if norm==0:
        return None

    finger_dir /= norm

    nearest_object = None

    objects = []
    nearest_distance = float("inf")

    for box in boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()

        center = np.array([(x1+x2)/2,(y1+y2)/2])

        obj_vec = center - fingertip

        distance = np.linalg.norm(obj_vec)

        if distance==0:
            continue

        obj_dir = obj_vec/distance

        dot = np.dot(finger_dir,obj_dir)

        if dot <= 0:
            continue

        dot = np.clip(dot,-1.0,1.0)
        angle = np.degrees(np.arccos(dot))

        if angle <= max_angle:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
                        # objects.append(
                        #     {
                        #         "class": result.names[class_id],
                        #         "confidence": round(confidence, 3),
                        #         "bbox": {
                        #             "x1": int(x1),
                        #             "y1": int(y1),
                        #             "x2": int(x2),
                        #             "y2": int(y2),
                        #         }
                        #     }
            nearest_object={
                    "class": result.names[class_id],
                    "confidence": round(confidence, 3),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                },
            }
            objects.append(nearest_object)

            
    return objects

YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
JSON_PATH = "src/output/vision_data.json"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150


def detections_to_text(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        vision_data = json.load(file)
        
    objects = vision_data["objects"]

    if len(objects) == 0:
        return "현재 탐지된 객체가 없습니다."

    sentences = []

    for index, obj in enumerate(objects, start=1):
        sentence = f"{index}번 객체는 {obj['class']}이며, confidence는 {obj['confidence']:.2f}입니다."

        sentences.append(sentence)

    return "\n".join(sentences)


yolo = YOLO(YOLO_MODEL_PATH)

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()


print("q : 종료")
print("l : 현재 YOLO 탐지 결과를 Gemma에게 전달")


def is_point_in_box(point,box):
    px,py = point
    x1,y1,x2,y2 = box

    return x1<=px<=x2 and y1<=py<=y2

while True:
    ret, frame = cap.read()

    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    #frame = cv2.flip(frame, 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 프레임 내 손 탐지
    hand = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    key = cv2.waitKey(1) & 0xFF

    if not ret:
        break
    if key == ord("q"):
        break

    height, width = frame.shape[:2]

    finger1 = (0,0)
    prev_finger1 = (0,0)

    # 탐지 결과의 각 손마다 선과 점 그리기
    for finger in hand.hand_landmarks:
        h, w = frame.shape[:2]  # 프레임 높이와 너비
        points = [(int(p.x * w), int(p.y * h)) for p in finger]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표

        # landmark를 연결하는 선 (skeleton) 그리기
        for c in connections:
            cv2.line(frame, points[c.start], points[c.end], (0,255,0), 2)

        # 각 관절 (landmark)에 점 그리기 (손가락 끝은 빨간 점, 그 외에는 파란 점)
        for i, point in enumerate(points):
            color = (0,0,255) if i in finger_tips else (255,0,0)
            cv2.circle(frame, point, 6 if i in finger_tips else 4, color, -1)
        finger1 = points[8]
        prev_finger1 = points[7]

    

    # 1. 현재 프레임 YOLO 탐지
    results = yolo.predict(
        source=frame,
        conf=0.25,
        iou=0.5,
        verbose=False,
    )

    result = results[0]


    # 2. YOLO 결과 화면 출력
    output_frame = result.plot()
    cv2.imshow("YOLO + Gemma", output_frame)


    # 3. YOLO 탐지 결과를 Dictionary 형태로 변환
    objects = get_nearest_pointed_object(finger1,prev_finger1,result.boxes,result)

    # for box in result.boxes:
    #         x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
    #         check_box=(x1,y1,x2,y2)
            # if is_point_in_box(finger1,check_box):
            #     class_id = int(box.cls[0].item())
            #     confidence = float(box.conf[0].item())
            #     objects.append(
            #         {
            #             "class": result.names[class_id],
            #             "confidence": round(confidence, 3),
            #             "bbox": {
            #                 "x1": int(x1),
            #                 "y1": int(y1),
            #                 "x2": int(x2),
            #                 "y2": int(y2),
            #             }
            #         }
    #             )
    vision_dict = {
        "image_width": width,
        "image_height": height,
        "objects": objects,
    }


    # 4. Dictionary → JSON 파일 저장
    if key == ord("l"):
        with open(JSON_PATH, "w", encoding="utf-8") as file:
            json.dump(vision_dict, file, ensure_ascii=False, indent=4)
        
        print("\n[JSON 파일 저장 완료]")

        # 5. JSON 파일 → Natural Language 변환
        vision_text = detections_to_text(JSON_PATH)
        print("\n[Vision Context]")
        print(vision_text)

        # 6. 탐지 결과를 Gemma에 Context로 전달
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """
                                Instruction:
                                주어진 객체 탐지 정보를 바탕으로 사용자의 요구에 가장 부합하는 객체를 고르시오.

                                Constraint:
                                탐지 결과에 없는 객체를 추측하지 마시오.

                                Output Format:
                                고른 객체와 요구에 따른 출력.
                                한국어 2문장 이내.
                               """
                },
                {
                    "role": "user",
                    "content": f"""
                                Context:
                                {vision_text},
                                물건을 전달해줘
                               """
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )

        answer = response["choices"][0]["message"]["content"]

        print("\n[Gemma]")
        print(answer)
        #tts.tts(answer)


cap.release()
cv2.destroyAllWindows()