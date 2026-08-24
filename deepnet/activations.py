import math

class ReLU:
    def forward(self, x):
        self.input = x
        return [max(0.0, val) for val in x]

    def backward(self, grad_output, learning_rate=None):
        return [g if self.input[i] > 0 else 0.0 for i, g in enumerate(grad_output)]


class Sigmoid:
    def forward(self, x):
        self.output = [1.0 / (1.0 + math.exp(-max(min(val, 500), -500))) for val in x]
        return self.output

    def backward(self, grad_output, learning_rate=None):
        return [grad_output[i] * self.output[i] * (1 - self.output[i]) for i in range(len(grad_output))]


class Tanh:
    def forward(self, x):
        self.output = [math.tanh(val) for val in x]
        return self.output

    def backward(self, grad_output, learning_rate=None):
        return [grad_output[i] * (1 - self.output[i] ** 2) for i in range(len(grad_output))]


class Softmax:
    def forward(self, x):
        max_val = max(x)
        exps = [math.exp(val - max_val) for val in x]
        sum_exps = sum(exps)
        self.output = [e / sum_exps for e in exps]
        return self.output

    def backward(self, grad_output, learning_rate=None):
        # Simplified for educational purposes
        return grad_output
