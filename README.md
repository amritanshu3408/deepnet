# DeepNet 🧠

**Neural Networks from Scratch in Pure Python**

A clean, educational, and fully functional neural network library built without any machine learning frameworks (no TensorFlow, no PyTorch, no NumPy dependency for core math in basic mode).

Perfect for understanding how neural networks actually work under the hood.

## Features

- Dense (Fully Connected) layers
- Activation functions: ReLU, Sigmoid, Tanh, Softmax
- Loss functions: MSE, Cross-Entropy
- Optimizers: SGD, Momentum
- Mini-batch training support
- Model save & load
- Clean and readable code

## Installation

```bash
git clone https://github.com/amritanshu3408/deepnet.git
cd deepnet
python examples/train_xor.py
```

## Quick Example

```python
from deepnet.network import NeuralNetwork
from deepnet.layers import Dense
from deepnet.activations import ReLU, Sigmoid

model = NeuralNetwork([
    Dense(2, 4),
    ReLU(),
    Dense(4, 1),
    Sigmoid()
])

model.train(X, y, epochs=5000, learning_rate=0.1)
predictions = model.predict(X)
```

## Project Structure

```
deepnet/
├── deepnet/
│   ├── __init__.py
│   ├── network.py
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   └── optimizers.py
├── examples/
│   └── train_xor.py
└── README.md
```

## Why this project?

Most people use PyTorch or TensorFlow without understanding the fundamentals.  
This library forces you to understand:

- Forward propagation
- Backpropagation
- Gradient calculation
- Weight updates

Great for interviews, learning, and GitHub profile strength.

---

Built with ❤️ by Amritanshu
