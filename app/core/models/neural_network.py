import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.momentum_weights = np.zeros_like(self.weights)
        self.momentum_bias = np.zeros_like(self.bias)
        self.rmsprop_weights = np.zeros_like(self.weights)
        self.rmsprop_bias = np.zeros_like(self.bias)
        self.t = 0  # Time step for bias correction

    def forward(self, X):
        self.input = X
        output = np.dot(X, self.weights) + self.bias
        if self.activation == 'relu':
            self.output = np.maximum(0, output)
        else:
            self.output = output
        return self.output

    def backward(self, gradient):
        if self.activation == 'relu':
            gradient = gradient * (self.output > 0)
        self.gradient_weights = np.dot(self.input.T, gradient)
        self.gradient_bias = np.sum(gradient, axis=0)
        gradient_input = np.dot(gradient, self.weights.T)
        return gradient_input


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            for batch_start in range(0, len(X), batch_size):
                batch_end = batch_start + batch_size
                X_batch = X[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]
                self.train_on_batch(X_batch, y_batch)

            output = self.predict(X)
            loss = np.mean((output - y) ** 2)

            # Print the epoch and loss in the desired format
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{len(X) // batch_size}/{len(X) // batch_size} [==============================] - {loss:.4f}")

    def train_on_batch(self, X, y):
        # Forward pass
        output = X
        for layer in self.layers:
            output = layer.forward(output)

        # Backpropagation
        gradient = 2 * (output - y)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        # Update weights and biases using Adam optimizer
        for layer in self.layers:
            self.update_layer(layer)

    def update_layer(self, layer):
        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.99
        epsilon = 1e-8

        # Update momentum
        layer.momentum_weights = beta1 * layer.momentum_weights + (1 - beta1) * layer.gradient_weights
        layer.momentum_bias = beta1 * layer.momentum_bias + (1 - beta1) * layer.gradient_bias

        # Update RMSProp
        layer.rmsprop_weights = beta2 * layer.rmsprop_weights + (1 - beta2) * (layer.gradient_weights ** 2)
        layer.rmsprop_bias = beta2 * layer.rmsprop_bias + (1 - beta2) * (layer.gradient_bias ** 2)

        # Bias correction
        layer.t += 1
        momentum_weights_corrected = layer.momentum_weights / (1 - beta1 ** layer.t)
        momentum_bias_corrected = layer.momentum_bias / (1 - beta1 ** layer.t)

        rmsprop_weights_corrected = layer.rmsprop_weights / (1 - beta2 ** layer.t)
        rmsprop_bias_corrected = layer.rmsprop_bias / (1 - beta2 ** layer.t)

        # Update weights and biases
        layer.weights -= learning_rate * (momentum_weights_corrected / (np.sqrt(rmsprop_weights_corrected) + epsilon))
        layer.bias -= learning_rate * (momentum_bias_corrected / (np.sqrt(rmsprop_bias_corrected) + epsilon))

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
