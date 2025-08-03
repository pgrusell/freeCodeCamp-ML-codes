#!/usr/bin/env python3

from db import Unpacker
import keras
import numpy as np

# Get the data
unpacker = Unpacker()


def create_data_set(data_set):
    '''
    Function to transform the mapped data from the unpacker to
    cal data, i.e, ready to be read using the keras functions
    '''

    n = len(data_set)
    return np.reshape(data_set, (n, -1))


# Create data sets
x_train_ds = create_data_set(unpacker.train)
y_train_ds = np.array(unpacker.train_label)

x_test_ds = create_data_set(unpacker.test)
y_test_ds = np.array(unpacker.test_label)

# Normalize the data
x_train_ds = x_train_ds.astype('float32') / 255.0
x_test_ds = x_test_ds.astype('float32') / 255.0

# Create the model
model = keras.Sequential()
model.add(keras.Input(shape=(len(x_train_ds[0]), )))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(len(set(y_train_ds)), activation='softmax'))

# Compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(x_train_ds, y_train_ds, batch_size=250, epochs=20)

# Testing
test_loss, test_acc = model.evaluate(x_test_ds, y_test_ds)

print(test_acc)
print(test_loss)
