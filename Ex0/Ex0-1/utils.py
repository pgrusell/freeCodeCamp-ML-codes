import numpy as np
import numpy.random as npr
from numba import njit

# Activation functions and their derivatives


@njit
def relu(x):
    return np.maximum(0, x)


@njit
def relu_diff(x):
    return 1.0 * (x > 0)


@njit
def tanh(x):
    return np.tanh(x)


@njit
def tanh_diff(x):
    t = np.tanh(x)
    return 1.0 - t * t


@njit
def compute_error(weights_col, next_errors, zLj):
    # Computes the backpropagated error for a given neuron
    w = np.ascontiguousarray(weights_col)
    e = np.ascontiguousarray(next_errors)
    return (w @ e) * tanh_diff(zLj)


class NN:
    '''
    This class implements methods for building a NN for regression (although can be easily adapted)
    to solve a classification problem. The characteristics of the network are:

    * Input layer with a neuron per dimension of the input data, e.g, (0.1, 0.2) -> 2 neurons
    * Output layer with a neuron per dimension of the output data.
    * The number of hidden layer can be set and also the number of neuron per layer, however they
      must have to be homogeneous (same number of neurons per layer). 
    '''

    def __init__(self, n_layers: int, n_neurons: int, x_data: np.ndarray, y_data: np.ndarray, activation='tanh') -> None:
        '''
        :n_layers: number of hidden layers
        :n_neurons: number of neurons per layer
        :x_data: input data. The N-dimensional input point
        :y_data: output data. The M-dimensional output point'
        '''
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        # Normalize the data
        self.x_mean, self.x_std = x_data.mean(axis=0), x_data.std(axis=0)
        x_data = (x_data - self.x_mean) / self.x_std

        self.y_mean, self.y_std = y_data.mean(), y_data.std()
        y_data = (y_data - self.y_mean) / self.y_std

        self.x_data = x_data
        self.y_data = y_data

        # Dimensions of the input and output layer
        _, self.Mx = x_data.shape
        _, self.Ny = y_data.shape

        # Choose activation function
        if activation == 'relu':
            self.activation_function = relu
            self.activation_function_diff = relu_diff
        else:
            self.activation_function = tanh
            self.activation_function_diff = tanh_diff

    def build(self) -> None:
        '''
        Method for building the matrices that represent the activations, weights and bias of
        every neuron.
        '''
        self.activations = []
        self.z = []

        # Initialize the layers
        # Input
        self.activations.append(np.zeros(self.Mx))
        self.z.append(np.zeros(self.Mx))
        # Hidden
        for _ in range(self.n_layers):
            self.activations.append(np.zeros(self.n_neurons))
            self.z.append(np.zeros(self.n_neurons))
        # Output
        self.activations.append(np.zeros(self.Ny))
        self.z.append(np.zeros(self.Ny))

        # Weights and bias
        self.weights = []
        self.bias = []

        for i in range(self.n_layers + 1):
            m = len(self.activations[i])
            n = len(self.activations[i + 1])

            # He initialization
            self.weights.append(npr.normal(0, np.sqrt(2 / m), (n, m)))
            self.bias.append(np.zeros(n))  # Initialize the bias to 0

    def forward_propagation(self, x_point: np.ndarray) -> None:
        '''
        This method takes one point of the x_data as input and calculates the activation of every
        neuron given the current weights.
        '''
        self.activations[0] = x_point

        for i in range(self.n_layers + 1):
            zi = self.weights[i] @ self.activations[i] + self.bias[i]
            self.z[i + 1] = zi

            if i == self.n_layers:  # linear output in the last layer
                self.activations[i + 1] = zi
            else:
                self.activations[i + 1] = self.activation_function(zi)

    def initialize_increases(self) -> None:
        '''
        Method for restoring/initializing the weights/bias increases for the gradient
        descent method.
        '''
        self.delta_weights = [np.zeros_like(w) for w in self.weights]
        self.delta_bias = [np.zeros_like(b) for b in self.bias]
        self.errors = [np.zeros_like(a) for a in self.activations]

    def back_propagation(self, y_point: np.ndarray) -> None:
        '''
        This method, given one expected output point (y_point) will evaluate the
        variation on the weights and bias and store it. Some notation:

        :L: total number of layers with weights
        :aLj: activation of the neuron j on the layer L
        :wLji: element of the matrix of weights that connects the neuron j of the layer
               L with the neuron i of the layer j
        :zLj: weighted entrance of the neuron j on the layer L
        :errorLj: j-th component of the gradient of the loss function with respect to the
                  zLj coordinate.
        '''
        L = self.n_layers + 1

        for l in reversed(range(1, L + 1)):
            J = len(self.activations[l])

            for j in range(J):
                zLj = self.z[l][j]
                aLj = self.activations[l][j]

                if l == L:  # Output layer
                    yj = y_point[j]
                    errorLj = 2 * (aLj - yj)
                else:
                    errorLj = compute_error(
                        self.weights[l][:, j], self.errors[l + 1], zLj)

                self.errors[l][j] = errorLj

                grad = errorLj * self.activations[l - 1]
                self.delta_weights[l - 1][j, :] += grad
                self.delta_bias[l - 1][j] += errorLj

    def mse(self, y_data: np.ndarray) -> float:
        '''
        Function to calculate the loss function given an expected data
        for a NN activated by its input.
        '''
        a = self.activations[-1]
        return np.mean((a - y_data) ** 2)

    def train(self, obj: float, eta: float, batch_size: int = 256) -> None:
        '''
        Given an initial data set x_data and y_data (initialized as class members)
        this function will update the weights minimizing the loss function using
        the gradient descend method.

        :C: Loss function, mse in our case
        :obj: If r = |C_new - C_old| / |C_old|, the method stops when r == obj
        :eta: Rate of change of the weights updated with the backpropagation method.
        '''
        X = self.x_data
        Y = self.y_data
        r = obj + 1
        epoch = 0

        while True:
            indices = np.arange(len(X))
            npr.shuffle(indices)
            X, Y = X[indices], Y[indices]

            C_old = 0
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                self.initialize_increases()

                for x, y in zip(X[start:end], Y[start:end]):
                    self.forward_propagation(x)
                    self.back_propagation(y)
                    C_old += self.mse(y)

                for l in range(len(self.weights)):
                    self.delta_weights[l] /= batch_size
                    self.delta_bias[l] /= batch_size
                    self.weights[l] -= eta * self.delta_weights[l]
                    self.bias[l] -= eta * self.delta_bias[l]

            C_old /= len(X)

            C_new = 0
            for x, y in zip(X, Y):
                self.forward_propagation(x)
                C_new += self.mse(y)
            C_new /= len(X)

            r = abs(C_new - C_old) / (abs(C_old) + 1e-8)
            epoch += 1
            print(f"Epoch {epoch:03d} | Loss: {C_new:.6f} | r: {r:.4e}")
            if r <= obj:
                print(f"Train converged at {epoch} epochs with r = {r:.4e}")
                break

    def predict(self, x: np.ndarray) -> np.ndarray:
        '''
        Method to make predictions, can handle both single point and batch of inputs.
        '''
        if x.ndim == 1:
            x = (x - self.x_mean) / self.x_std
            self.forward_propagation(x)
            return self.activations[-1] * self.y_std + self.y_mean
        else:
            x = (x - self.x_mean) / self.x_std
            y_preds = []
            for point in x:
                self.forward_propagation(point)
                y_preds.append(self.activations[-1] * self.y_std + self.y_mean)
            return np.array(y_preds)
