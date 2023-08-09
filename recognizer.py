import cv2
import numpy as np
from keras.models import Model


class DigitRecognizer:
    """
    숫자 인식기 클래스입니다.
    """

    def __init__(self, model: Model):
        """
        숫자 인식기 생성자입니다.
        :param model: MNIST로 학습된 Keras 모델
        """
        self.model = model

    def run(self, gray_image: np.ndarray) -> np.ndarray:
        """
        주어진 이미지에서 숫자를 인식합니다.
        :param gray_image: 숫자가 쓰여진 gray 이미지
        :return: 각 숫자에 대한 예측 확률을 담은 배열
        """
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

        if __debug__:
            cv2.imshow('digit', cv2.resize(x, (112, 112)))

        x = x.reshape((1, 28, 28, 1))
        probabilities = self.model.predict(x, verbose=0)
        assert probabilities.shape == (1, 10)

        return probabilities[0]
