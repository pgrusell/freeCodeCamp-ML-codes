import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import matplotlib.pyplot as plt
from utils import NN

# Generate some random data
x_data = npr.uniform(-10, 10, (1000, 2))
y_data = npl.norm(x_data, axis=1, keepdims=True)

# Build the NN
neural_network = NN(n_layers=4, n_neurons=32,
                    x_data=x_data, y_data=y_data)
neural_network.build()
neural_network.train(obj=0.01, eta=0.00001)

# Testing
x_test = npr.uniform(-10, 10, (200, 2))
y_true = npl.norm(x_test, axis=1, keepdims=True)

# Prediction
y_pred = np.array([neural_network.predict(x) for x in x_test])
y_pred = y_pred.reshape(-1, 1)

# Cálculo error
mse = np.mean((y_pred - y_true) ** 2)
print(f"Test MSE: {mse:.6f}")

# Statistical analysis
# Correlation (straight line)
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())

plt.figure(figsize=(8, 6))
plt.plot(y_true, y_pred, 'o', alpha=0.5)
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Real value")
plt.ylabel("Predicted value")
plt.title("Correlation")
plt.grid(True)
plt.show()

# Error distribution (Gaussian)
error = y_pred - y_true

plt.figure(figsize=(8, 6))
plt.hist(y_pred - y_true, bins=np.linspace(error.min(), error.max(),
         20), color='blue', edgecolor='black', alpha=0.4, label=f'mean = {error.mean() : .3f}\nstd = {error.std() : .3f}')
plt.title("Error distribution")
plt.xlabel("Error")
plt.ylabel("# events")
plt.legend()
plt.show()
