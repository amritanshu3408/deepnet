import random

class Dense:
    """Fully connected layer."""

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Xavier-like initialization
        limit = (6 / (input_size + output_size)) ** 0.5
        self.weights = [[random.uniform(-limit, limit) for _ in range(output_size)] for _ in range(input_size)]
        self.biases = [0.0] * output_size
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = []
        for j in range(self.output_size):
            val = self.biases[j]
            for i in range(self.input_size):
                val += x[i] * self.weights[i][j]
            self.output.append(val)
        return self.output

    def backward(self, grad_output, learning_rate):
        grad_input = [0.0] * self.input_size

        for i in range(self.input_size):
            for j in range(self.output_size):
                grad_input[i] += grad_output[j] * self.weights[i][j]
                # Update weights
                self.weights[i][j] -= learning_rate * grad_output[j] * self.input[i]

        for j in range(self.output_size):
            self.biases[j] -= learning_rate * grad_output[j]

        return grad_input
