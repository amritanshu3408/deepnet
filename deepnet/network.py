class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(len(X)):
                # Forward
                output = self.forward(X[i])

                # Compute MSE loss gradient
                grad = [output[j] - y[i][j] for j in range(len(output))]
                loss = sum(g ** 2 for g in grad) / len(grad)
                total_loss += loss

                # Backward
                self.backward(grad, learning_rate)

            if verbose and (epoch + 1) % 500 == 0:
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    def predict(self, X):
        return [self.forward(x) for x in X]
