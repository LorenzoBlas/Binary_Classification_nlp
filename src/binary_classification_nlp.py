from keras.datasets import imdb
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos IMDB (10,000 palabras más frecuentes)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Función para convertir secuencias en vectores binarios
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Vectorizar datos de entrenamiento y prueba
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Convertir etiquetas a float32
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Pérdida durante entrenamiento y validación
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
print(f"Loss: {results[0]}, Accuracy: {results[1]}")

predictions = model.predict(x_test[:2])
print(predictions)


