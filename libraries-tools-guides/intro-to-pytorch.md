# Intro to PyTorch


## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Tensors: The Building Blocks](#tensors-the-building-blocks)
4. [Basic Operations](#basic-operations)
5. [Indexing, Slicing, and Joining](#indexing-slicing-and-joining)
6. [Computational Graphs and Autograd](#computational-graphs-and-autograd)
7. [Building a Simple Neural Network](#building-a-simple-neural-network)
8. [Training the Neural Network](#training-the-neural-network)
9. [Conclusion](#conclusion)


---

PyTorch, a product of Facebook's AI Research lab, is an open-source machine learning library known for its dynamic computational graph and flexibility, making it a favorite for deep learning. This guide will delve into the core essentials of PyTorch, covering basic operations, tensor manipulations, and constructing a simple neural network.

---

## Introduction

In PyTorch, the fundamental unit is the tensor. A tensor is a multi-dimensional array, much like arrays in NumPy, with additional capabilities for GPU-accelerated computation. This is crucial for performing high-level mathematical functions and operations on large datasets, a common requirement in data science, machine learning, and deep learning.

## Installation

To begin, install PyTorch on your machine. You can do this via pip or Anaconda with the following command:

```bash
pip install torch torchvision

```

## Tensors: The Building Blocks

Tensors in PyTorch are multi-dimensional arrays. They are the basic units used to encapsulate data which the model will learn from. They are crucial as they hold the data (inputs, outputs, parameters) that flows through the neural network.

```python
import torch

# Creating a tensor from a list
tensor_from_list = torch.tensor([[1, 2], [3, 4]])

# Creating a tensor of ones with dimensions 2x3
ones_tensor = torch.ones(2, 3)

# Creating a tensor of zeros with dimensions 2x3
zeros_tensor = torch.zeros(2, 3)

# Creating a random tensor with dimensions 2x3
random_tensor = torch.rand(2, 3)

```

In the code snippet above, we created tensors using different functions provided by PyTorch. The `torch.tensor` function creates a tensor from a list, `torch.ones` and `torch.zeros` functions create tensors filled with ones and zeros, respectively, and `torch.rand` function creates a tensor with random values between 0 and 1.

## Basic Operations

PyTorch supports a wide range of operations on tensors, akin to operations on NumPy arrays, but with additional optimizations for GPU computation. These operations include arithmetic operations which are usually performed element-wise (i.e., operations are carried out element by element).

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Element-wise addition
c = a + b  # Equivalent to torch.add(a, b)

# Element-wise subtraction
d = a - b  # Equivalent to torch.sub(a, b)

# Element-wise multiplication
e = a * b  # Equivalent to torch.mul(a, b)

# Element-wise division
f = a / b  # Equivalent to torch.div(a, b)

```

In the code snippet above, tensors `a` and `b` are subjected to basic arithmetic operations. These operations are performed element-wise, which means each element in `a` is operated with the corresponding element in `b`.

## Indexing, Slicing, and Joining

Manipulating tensors by indexing, slicing, or joining is a common practice in PyTorch, which is essential for data preprocessing and manipulation. These operations are similar to those in NumPy but optimized for tensor operations.

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Indexing
element = a[0, 1]  # Retrieves the element at row 0, column 1

# Slicing
row = a[1, :]  # Retrieves the second row

# Joining
joined_tensor = torch.cat((a, a), dim=0)  # Joins 'a' with 'a' along rows

```

Here, `torch.cat` is used to concatenate two tensors along a specified dimension, which is essential for merging datasets or outputs from multiple neural network layers.

## Computational Graphs and Autograd

One of PyTorch's core features is its ability to build dynamic computational graphs and perform automatic differentiation using the Autograd module. A computational graph is a directed acyclic graph where nodes correspond to operations or variables. Autograd provides automatic differentiation for all operations on tensors, which is crucial for backpropagation during training of neural networks.

```python
x = torch.tensor(1.0, requires_grad=True)
y = x * 2
z = y ** 2
z.backward()  # Computes the gradients
print(x.grad)  # Outputs '8.0'

```

In this code snippet, a simple computational graph is created. The `requires_grad` attribute tells PyTorch to track operations on `x`, so the gradient can be computed during backpropagation.

## Building a Simple Neural Network

Constructing neural networks in PyTorch is straightforward, thanks to the `torch.nn` module. Below is a simple example:

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 2)  # A linear layer with 3 inputs and 2 outputs

    def forward(self, x):
        return self.fc1(x)

# Instantiate and use the network
net = SimpleNN()
input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
output_tensor = net(input_tensor)

```

In the code above, a simple neural network is defined with one linear layer. The `forward` method defines the forward pass of the network, dictating how the input data flows through the network.

## Training the Neural Network

Training a neural network involves defining a loss function, an optimization algorithm, and iterating through the training data to minimize the loss.

```python
# Defining the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()  # Zero the gradients
    output = net(input_tensor)
    loss = criterion(output, torch.tensor([[0.0, 1.0]]))  # Assuming some target values
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

```

In the training loop, `optimizer.zero_grad()` is called to clear the old gradients because by default, gradients in PyTorch accumulate. The `loss.backward()` computes the gradient of the loss with respect to the parameters, and `optimizer.step()` performs a parameter update based on the current gradient.

## Conclusion

This guide sought to

provide a detailed introduction to the core essentials of PyTorch, transitioning from basic tensor operations to constructing and training a simple neural network. The dynamic nature and ease of use make PyTorch a powerful tool for deep learning research and applications.

---
