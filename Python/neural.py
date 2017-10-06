import cv2
import numpy as np
from PIL import Image
from mnist import MNIST
from os.path import exists
from json import load, dump

trained_data_dir = "./data/trained_data.json"
train_MNIST_data_dir = "./data/MNIST_train_data/"
color_part = 0.45
frames_multiplier = 1.5


class Neural:
    def __init__(self):
        self.coefficients = np.ones((10, 784), np.float32)

    def analyze(self, image_dir: str):
        image = self.__preprocessor(image_dir)
        results = np.zeros(10, np.float32)
        for i in range(len(results)):
            results[i] = np.sum(np.dot(self.coefficients[i], image))
        return np.argmax(results)

    def train(self, retrain=False):
        if not retrain and exists(trained_data_dir):
            self.coefficients = np.array(load(open(trained_data_dir, 'r', encoding="UTF-8")), np.float32)
            return
        train_images, train_labels = MNIST(train_MNIST_data_dir).load_training()
        train_images = self.__normalize(train_images)
        for i in range(len(train_labels)):
            image = train_images[i]
            label = train_labels[i]
            prediction = self.__analyze_train_image(image)
            if prediction != label:
                self.coefficients[label] += image
                self.coefficients[prediction] -= image
        dump(self.coefficients.tolist(), open(trained_data_dir, 'w', encoding="UTF-8"))

    def __analyze_train_image(self, image: np.array):
        results = np.zeros(10, np.float32)
        for i in range(len(results)):
            results[i] = np.sum(np.dot(self.coefficients[i], image))
        return np.argmax(results)

    def __preprocessor(self, image_dir: str):
        image = Image.open(image_dir).convert("L")
        image = image.point(self.__black_white_filter(np.asarray(image), image_dir[-3:].lower() == "jpg"))
        return self.__convert_to_square(np.asarray(image))

    def __convert_to_square(self, image: np.array):
        image = self.__crop(image)
        max_side_size = int(max(len(image), len(image[0])) * frames_multiplier)

        vertical_indent = (max_side_size - len(image)) // 2  # Находим вертикальный отступ
        horizontal_indent = (max_side_size - len(image[0])) // 2  # Находим горизонтальный отступ

        new_image = np.zeros((max_side_size, max_side_size), np.float32)
        for i in range(len(image)):
            for j in range(len(image[i])):
                new_image[i + vertical_indent][j + horizontal_indent] = image[i][j]
        return cv2.resize(new_image, (28, 28)).ravel()

    def __black_white_filter(self, image: np.array, jpg: bool):
        if jpg:
            threshold = 255 - self.__avg_color(image)

            def color_gen(x):
                return 1 if x < threshold else 0

            return color_gen
        else:
            threshold = self.__avg_color(image)

            def color_gen(x):
                return 0 if x < threshold else 1

            return color_gen

    def __crop(self, image: np.array):
        crop_coordinates = [None, None, None, None]  # y1, x1, y2, x2
        img_width = len(image[0]) - 1
        img_height = len(image) - 1
        for i in range(len(image[0])):
            if crop_coordinates[1] is None and self.__check_column(image, i):
                crop_coordinates[1] = i
            if crop_coordinates[3] is None and self.__check_column(image, img_width - i):
                crop_coordinates[3] = img_width - i
            if crop_coordinates[1] is not None and crop_coordinates[3] is not None:
                break
        for i in range(len(image)):
            if crop_coordinates[0] is None and self.__check_row(image, i):
                crop_coordinates[0] = i
            if crop_coordinates[2] is None and self.__check_row(image, img_height - i):
                crop_coordinates[2] = img_height - i
            if crop_coordinates[0] is not None and crop_coordinates[2] is not None:
                break
        return image[crop_coordinates[0]:crop_coordinates[2], crop_coordinates[1]: crop_coordinates[3]]

    @staticmethod
    def __check_row(matrix: np.array, index: int):
        for i in range(len(matrix[index])):
            if matrix[index][i] == 1:
                return True
        return False

    @staticmethod
    def __check_column(matrix: np.array, index: int):
        for i in range(len(matrix)):
            if matrix[i][index] == 1:
                return True
        return False

    @staticmethod
    def __normalize(array: list):
        for i in range(len(array)):
            array[i] = np.array(array[i]) / 255
        return array

    @staticmethod
    def __avg_color(image: np.array):
        return int(np.sum(image) / (len(image) * len(image[0])) * color_part) + 1


if __name__ == "__main__":
    neural = Neural()
    neural.train()

    samples_dir = "./samples/"
    samples_list = ["eleven.jpg", "four.png", "one_1.png", "one_2.png", "seven_1.png", "seven_2.jpg", "six_1.png",
                    "six_2.jpg", "six_3.jpg"]
    for sample in samples_list:
        print(neural.analyze(samples_dir + sample))
