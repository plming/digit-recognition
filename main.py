import cv2
import numpy as np
from keras.models import load_model

from detector import detect_digit
from recognizer import DigitRecognizer

if __name__ == '__main__':
    print("사용법")
    print("q: 종료")

    model = load_model('model', compile=False)
    recognizer = DigitRecognizer(model)

    capture = cv2.VideoCapture(1)
    if not capture.isOpened():
        raise Exception("Could not open video device")

    UI_COLOR = (0, 0, 0xFF)
    while True:
        has_read, frame = capture.read()
        if not has_read:
            continue

        bounding_box = detect_digit(frame)
        if bounding_box is not None:
            cropped = bounding_box.crop(frame)

            pred = recognizer.run(cropped)
            digit = np.argmax(pred)
            probability = pred[digit]

            cv2.rectangle(frame, bounding_box.left_top, bounding_box.right_bottom, UI_COLOR, 2)
            cv2.putText(frame, f"{digit} ({probability:.2f})", (bounding_box.x, bounding_box.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, UI_COLOR, 2)

        cv2.imshow('out', frame)

        cmd = cv2.waitKey(1)
        if cmd == ord('q'):
            break
