import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib as plt
import numpy as np

#Import Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist

#Load dataset, split data into training and validation sets, scale X
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#Rename classes to IRL categories
class_names = ["T-shirt/top", "Pants", "Sweater", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

#Create model using sequential API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))
keras.utils.plot_model(model)

#Compute loss
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

#Train model
history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid))

#Plot accuracy, loss using pandas
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1) #Set vertical range to [0,1]
plt.show()