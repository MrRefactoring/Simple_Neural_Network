import cv2
import numpy as np
from PIL import Image
from mnist import MNIST


class Neural:
    def __init__(self):
        self.frame = (700, 700)  # 500 на 500
        self.coefficients = np.ones((10, 784), np.float32) - .99

    def analyze(self, image_directory):
        """
        :param image_directory: путь до картинки, которую необходимо загрузить
        :return: предсказание нейросети
        """
        results = np.zeros(10, np.float32)  # Массив из нулей, который хранит значения анализа для каждой из 10 цифр
        image = self.__preprocessor(image_directory)  # Загружаем картинку

        for i in range(len(results)):
            results[i] = np.sum(np.dot(self.coefficients[i], image))
        return np.argmax(results)

    def analyze_mnist(self, image):
        results = np.zeros(10, np.float32)
        for i in range(len(results)):
            results[i] = np.sum(np.dot(self.coefficients[i], image))
        return np.argmax(results)

    def antiravel(self, image):
        array = list()
        for i in range(len(image) // 28):
            array.append(image[28 * i: 28 * i + 28])
        return array

    def train(self):
        images, labels = MNIST("./train_data/").load_training()
        images = self.__normalize(images)  # Проводим нормализацию изображений
        for i in range(len(labels)):
            image = images[i]
            label = labels[i]
            prediction = self.analyze_mnist(image)
            if prediction != label:
                self.coefficients[prediction] -= image
                self.coefficients[label] += image

    def __preprocessor(self, image_directory):
        """
        :param image_directory: путь до картинки, которую необходимо обработать
        :return: возвращает плоский numpy array
        """
        # Читаем картинку из файла в image, после чего пропускаем ее через ЧБ фильтр
        if image_directory[-3:].lower() == "jpg":
            image = Image.open(image_directory).convert("L").point(self.__bw_filter_jpg(254))
        else:
            image = Image.open(image_directory).convert("L").point(self.__bw_filter(50))
        #image = Image.open(image_directory).convert("L").point(self.__bw_filter(50))
        return self.__square(np.asarray(image))

    @staticmethod
    def __bw_filter(threshold=127):  # Черно-белый фильтр с порогом threshold
        """
        :param threshold: пороговое значение, после которого выставляется белый цвет
        :return: возвращает ф-ю, которая вызываестся внутри PIL
        """
        def table_gen(x):
            return 0 if x < threshold else 255
        return table_gen

    @staticmethod
    def __bw_filter_jpg(threshold=127):  # Черно-белый фильтр с порогом threshold для изображений формата JPG
        """
        :param threshold: пороговое значение, после которого выставляется белый цвет
        :return: возвращает ф-ю, которая вызываестся внутри PIL
        """

        def table_gen(x):
            return 255 if x < threshold else 0

        return table_gen

    @staticmethod
    def __normalize(array):
        for i in range(len(array)):
            array[i] = np.array(array[i], np.float32) / 255
        return array

    def __cut(self, image):
        sp = []
        ep = []
        flag = False
        for i in range(len(image)):
            for j in range(len(image[i])):
                if image[i][j] > 0:
                    sp.append(i)
                    flag = True
                    break
            if flag:
                break
        flag = False
        for i in list(range(len(image)))[::-1]:
            for j in list(range(len(image[i])))[::-1]:
                if image[i][j] > 0:
                    ep.append(i)
                    flag = True
                    break
            if flag:
                break
        for i in range(len(image[0])):
            if self.get_vertical(image, i):
                sp.append(i)
                break
        for i in list(range(len(image[0])))[::-1]:
            if self.get_vertical(image, i):
                ep.append(i)
                break
        cv2.imshow("", image[sp[0]:ep[0], sp[1]:ep[1]])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image[sp[0]:ep[0], sp[1]:ep[1]]

    @staticmethod
    def get_vertical(image, index):
        for i in range(len(image)):
            if image[i][index] > 0:
                return True
        return False

    def __square(self, image):
        """
        Метод для преобразования картинки в одномерный массив, размером 784.
        Сначала просто переводим до размера NxN, потом resize до 28x28
        После чего делаем массив плоским
        :param image: numpy массив, картинка
        :return: image as 784 1D array
        """
        image = self.__cut(image)
        max_side_size = int(max(len(image), len(image[0])) * 1.5)  # Ищем максимально большую сторону
        # Создаем новую картинку разммером max_size x max_side
        new_image = np.zeros((max_side_size, max_side_size), np.float32)
        vertical_indent = (max_side_size - len(image)) // 2  # Находим вертикальный отступ
        horizontal_indent = (max_side_size - len(image[0])) // 2  # Находим горизонтальный отступ
        # Преобразовываем картинку под нужный формат
        for i in range(len(image)):
            for j in range(len(image[i])):
                new_image[i + vertical_indent][j + horizontal_indent] = image[i][j]
        #print(cv2.resize(new_image, (28, 28)).tolist())
        cv2.imshow("Image", cv2.resize(new_image, (28, 28)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return cv2.resize(new_image, (28, 28)).ravel()  # Делаем resize новой картинки и возвращаем ее
