import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model

model = load_model('model.keras', compile=False)

capture = cv2.VideoCapture(1)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = capture.read()

    box_size = height // 2
    left_top = (width // 2 - box_size // 2, height // 2 - box_size // 2)
    right_bottom = (width // 2 + box_size // 2, height // 2 + box_size // 2)

    roi = frame[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()

    GREEN = (0, 255, 0)
    cv2.rectangle(frame, left_top, right_bottom, GREEN, 2)

    x = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (28, 28), interpolation=cv2.INTER_AREA)
    x = cv2.bitwise_not(x)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)
    digit = np.argmax(pred)
    print(f"인식된 숫자는 {digit}입니다.")

    # display (digit, percentage) on the frame
    cv2.putText(frame, f"{digit} ({pred[0][digit] * 100:.2f}%)", (left_top[0], left_top[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

    cv2.imshow('Video', frame)
    cv2.imshow('ROI', cv2.resize(x[0], (200, 200)))

    if cv2.waitKey(1) == ord('q'):
        break
