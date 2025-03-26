import numpy as np

def load_mnist_local(filename="mnist.npz"):
    """Загружает локальный MNIST из .npz файла"""
    data = np.load(filename)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    # Нормализация данных
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Преобразование в 4D-формат для модели
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test

