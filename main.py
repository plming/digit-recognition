from datetime import datetime

import cv2
import numpy as np
from keras.models import load_model

from recognizer import DigitRecognizer

if __name__ == '__main__':
    print("사용법")
    print("q: 종료 / s: 스크린샷")

    model = load_model('model-2023-08-04T09:51:44.880996.keras', compile=False)
    recognizer = DigitRecognizer(model)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise Exception("Could not open video device")

    UI_COLOR = [0, 0, 0xFF]
    while True:
        has_read, frame = capture.read()
        if not has_read:
            continue

        blur = cv2.GaussianBlur(frame, (9, 9), 0)

        # 숫자는 빨간색 글씨이므로, 해당 색상 영역을 마스킹함
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))

        morphology = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(morphology, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:  # 숫자를 찾았을 경우
            largest_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)

            pred = recognizer.run(mask[y:y + h, x:x + w])
            digit = np.argmax(pred)
            probability = pred[digit]

            cv2.putText(
                img=frame,
                text=f"{digit} ({probability * 100:.2f}%)",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=UI_COLOR,
                thickness=2
            )

            cv2.rectangle(frame, (x, y), (x + w, y + h), UI_COLOR, 2)

        cv2.imshow('frame', frame)

        cmd = cv2.waitKey(1)
        if cmd == ord('q'):
            break
        elif cmd == ord('s'):
            filename = f"screenshot-{datetime.utcnow().isoformat()}.jpg"
            cv2.imwrite(filename, frame)
            print(f"이미지가 {filename}에 저장되었습니다.")
