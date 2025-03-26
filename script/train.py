from data_loader import load_mnist_local  # Импортируем исправленную функцию
from models.autoencoder import create_autoencoder
import matplotlib.pyplot as plt

# Загрузка данных из локального MNIST
x_train, y_train, x_test, y_test = load_mnist_local("mnist.npz")

# Создание модели
autoencoder = create_autoencoder()

# Обучение модели
history = autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, x_test),
    verbose=1
)

# Сохранение модели
autoencoder.save("models/handwriting_autoencoder.h5")

# Визуализация потерь
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
