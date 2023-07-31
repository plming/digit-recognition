import cv2
import numpy as np
from keras.models import load_model

WIDTH, HEIGHT = 640, 480

model = load_model('model.keras', compile=False)

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    # TODO: 임시 해상도 조정을 위한 코드로 실 환경에선 제거해야 함
    frame = cv2.resize(src=frame, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    box_size = HEIGHT // 2
    left_top = (WIDTH // 2 - box_size // 2, HEIGHT // 2 - box_size // 2)
    right_bottom = (WIDTH // 2 + box_size // 2, HEIGHT // 2 + box_size // 2)

    GREEN = (0, 255, 0)
    cv2.rectangle(frame, left_top, right_bottom, GREEN, 2)

    roi = frame[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]

    x = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)
    print(f"인식된 숫자는 {np.argmax(pred)}입니다.")

    cv2.imshow('Video', frame)
    cv2.imshow('ROI', roi)
    cv2.imshow('Resized ROI', x[0])

    if cv2.waitKey(1) == ord('q'):
        break
