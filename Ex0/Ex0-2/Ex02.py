import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Build the dataset
x = np.random.uniform(-1, 1, (1000000, 2))
bg = np.random.normal(0, 0.2, (1000000, 1))
y = np.linalg.norm(x, axis=1, keepdims=True) + bg

# Normalization
x_mean = x.mean(axis=0, keepdims=True)
x_std = x.std(axis=0, keepdims=True)
x_norm = ((x - x_mean) / x_std)

y_mean = y.mean(axis=0, keepdims=True)
y_std = y.std(axis=0, keepdims=True)
y_norm = ((y - y_mean) / y_std)

# Architecture of the model
model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))

# Compilation
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Training
model.fit(x=x_norm, y=y_norm, batch_size=250, epochs=20)

# Testing
x_test = np.random.uniform(-1, 1, (1000, 2))
bg_test = np.random.normal(0, 0.2, (1000, 1))
y_test = np.linalg.norm(x_test, axis=1, keepdims=True) + bg_test

# Normalizing the test set
x_norm_test = (x_test - x_mean) / x_std
y_norm_test = (y_test - y_mean) / y_std

model.evaluate(x_norm_test, y_norm_test, batch_size=5, verbose=1)

# More complex analysis
y_pred_norm = model.predict(x_norm_test)
y_pred = y_pred_norm * y_std + y_mean

# Error estimation
error = y_pred - y_test

fig, ax = plt.subplots()
ax.set_title("Error estimation")
ax.hist(error, bins=np.linspace(-3, 3, 20), edgecolor='black',
        label=f'mean = {error.mean() : .3f}\nstd = {error.std() : .3f}')
ax.legend()

# Correlation estimation
min_val = min(y_pred.min(), y_test.min())
max_val = max(y_pred.max(), y_test.max())

corr, _ = pearsonr(y_pred.ravel(), y_test.ravel())

fig, ax = plt.subplots()
ax.set_title(f"Correlation estimation ({corr : .3f})")
ax.hist2d(y_pred.ravel(), y_test.ravel(), bins=(
    np.linspace(min_val, max_val, 100), np.linspace(min_val, max_val, 100)))
