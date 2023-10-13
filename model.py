# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:02:45 2023

@author: NENO
"""

#Para cargar y operar con el dataset
import pandas as pd
import numpy as np

#Para entrenar y crear el modelo de la red convolucional
import tensorflow as tf
from tensorflow.keras import layers, models

#Para imprimir el modelo
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# Define a normalization function
def normalize_column(matrix):
    norm = np.linalg.norm(matrix)
    if norm == 0:
        return matrix
    return matrix / norm

df = pd.read_json('dataset/MatrizPosiciones.json')

# Specify the percentage for the test set (30% in this case)
test_size = 0.3

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

X_train = tf.constant(train_df.drop("y",axis=1).values.tolist())
y_train = tf.constant(train_df.drop(["posiciones", "pawns", "knights","bishops", "rooks", "queens", "kings"],axis=1).values)


X_test = tf.constant(test_df.drop("y",axis=1).values.tolist())
y_test = tf.constant(test_df.drop(["posiciones", "pawns", "knights","bishops", "rooks", "queens", "kings"],axis=1).values)


#Creamos el modelo de la red convolucional

model = models.Sequential()
model.add(layers.Conv2D(128, (2, 2), activation='relu', input_shape=(7, 8, 8)))
model.add(layers.MaxPooling2D((1, 1)))
model.add(layers.Conv2D(256, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((1, 1)))
model.add(layers.Conv2D(256, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((1, 1)))
model.add(layers.Conv2D(256, (2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1))


#optimizer = tf.keras.optimizers.RMSprop(0.01)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#Entrenamos el modelo

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_train, y_train))


plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label = 'val_mae')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

metrics_df = pd.DataFrame(history.history)
metrics_df[["loss","val_loss"]].plot()


