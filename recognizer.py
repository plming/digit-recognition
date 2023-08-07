import cv2
import numpy as np
from keras.models import Model


class DigitRecognizer:
    def __init__(self, model: Model):
        self.model = model

    def run(self, gray_image: np.ndarray) -> np.ndarray:
        assert gray_image.ndim == 2

        # MNIST 데이터셋의 전처리 과정
        # 참고: http://yann.lecun.com/exdb/mnist/
        BASE_SIZE = (20, 20)
        base = cv2.resize(gray_image, BASE_SIZE, interpolation=cv2.INTER_AREA)

        BORDER_SIZE = 4
        x = cv2.copyMakeBorder(base, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                               cv2.BORDER_CONSTANT, value=0)

        # x값이 [0, 1] 범위에 있도록 scaling
        x_min = np.min(x)
        x_max = np.max(x)
        x = (x - x_min) / (x_max - x_min)

        x = x.reshape((1, 28, 28, 1))
        probabilities = self.model.predict(x)

        return probabilities[0]
