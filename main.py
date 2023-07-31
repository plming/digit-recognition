import cv2
import numpy as np
from keras.models import load_model

model = load_model('model.keras', compile=False)

capture = cv2.VideoCapture(1)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

box_size = height // 2
left, top = (width // 2 - box_size // 2, height // 2 - box_size // 2)
right, bottom = (width // 2 + box_size // 2, height // 2 + box_size // 2)

while True:
    ret, frame = capture.read()

    roi = frame[top:bottom, left:right]

    GREEN = (0, 255, 0)
    cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)

    x = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (28, 28), interpolation=cv2.INTER_AREA)
    x = cv2.bitwise_not(x)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)
    digit = np.argmax(pred)
    probability = pred[0][digit]

    cv2.putText(frame,
                f"{digit} ({probability * 100:.2f}%)",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                GREEN,
                2)

    cv2.imshow('frame', frame)
    cv2.imshow('x', cv2.resize(x[0], (200, 200)))

    if cv2.waitKey(1) == ord('q'):
        break
