from tensorflow.keras.models import load_model
from data_loader import load_mnist_local
import matplotlib.pyplot as plt

# Загрузка данных
x_train, y_train, x_test, y_test = load_mnist_local("C:/Users/user/u35hw2/data/mnist.npz")

# Загрузка обученной модели
autoencoder = load_model("C:/Users/user/u35hw2/models/handwriting_autoencoder.h5")

# Прогнозирование
decoded_imgs = autoencoder.predict(x_test)

# Визуализация
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Оригинал
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Сгенерированное
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Generated")
    plt.axis('off')

plt.show()
