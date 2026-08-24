"""Train a neural network on the classic XOR problem."""

from deepnet.network import NeuralNetwork
from deepnet.layers import Dense
from deepnet.activations import ReLU, Sigmoid

# XOR dataset
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [
    [0],
    [1],
    [1],
    [0]
]

model = NeuralNetwork([
    Dense(2, 8),
    ReLU(),
    Dense(8, 4),
    ReLU(),
    Dense(4, 1),
    Sigmoid()
])

print("Training on XOR problem...")
model.train(X, y, epochs=3000, learning_rate=0.3)

print("\nPredictions:")
for i, x in enumerate(X):
    pred = model.predict([x])[0][0]
    print(f"Input: {x} -> Predicted: {pred:.4f} | Target: {y[i][0]}")
