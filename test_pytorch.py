import torch
print(torch.__version__)

x = torch.empty(1) # 0D tensor
print("empty(1): ", x) # tensor([0.])

x = torch.empty(3) # 1D tensor
print("empty(3): ", x) # tensor([0., 0., 0.])

x = torch.empty(2, 3) # 2D tensor
print("empty(2, 3): ", x) # tensor([[0., 0., 0.], [0., 0., 0.]])

x = torch.empty(2, 3, 2) # 3D tensor
print("empty(2, 3, 2): ", x) # tensor([[[0., 0.], [0., 0.], [0., 0.]], [[0., 0.], [0., 0.], [0., 0.]]])

x = torch.rand(2, 3) # 2D tensor with random values
print("rand(2, 3): ", x) # tensor([[0.7676, 0.9067, 0.6828], [0.4913, 0.9732, 0.7139]])

x = torch.zeros(2, 3) # 2D tensor with zeros
print("zeros(2, 3): ", x) # tensor([[0., 0., 0.], [0., 0., 0.]])

x = torch.ones(2, 3) # 2D tensor with ones
print("ones(2, 3): ", x) # tensor([[1., 1., 1.], [1., 1., 1.]])

x = torch.tensor([2.5, 0.1]) # 1D tensor from list
print("tensor([2.5, 0.1]): ", x) # tensor([2.5000, 0.1000])

# GPU Support in PyTorch (CUDA) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([2.5, 0.1], device=device) # 1D tensor on GPU
print("tensor([2.5, 0.1], device=device): ", x) # tensor([2.5000, 0.1000], device='cuda:0')

# 2D tensor from list of lists
x = torch.tensor([[2, 3], [4, 5]]) # 2D tensor from list of lists
print("tensor([[2, 3], [4, 5]]): ", x) # tensor([[2, 3], [4, 5]])

# Create a tensor from a NumPy array
import numpy as np
x = np.array([2, 3])
x = torch.tensor(x) # 1D tensor from NumPy array
print("tensor(x): ", x) # tensor([2, 3])

x = np.array([[2, 3], [4, 5]])
x = torch.tensor(x) # 2D tensor from NumPy array
print("tensor(x): ", x) # tensor([[2, 3], [4, 5]])

# Create a tensor from another tensor
x = torch.tensor([2, 3])
y = torch.tensor(x) # 1D tensor from another tensor
print("tensor(x): ", x) # tensor([2, 3])

x = torch.tensor([[2, 3], [4, 5]])
y = torch.tensor(x) # 2D tensor from another tensor

# Create a tensor from a scalar
x = torch.tensor(3) # 0D tensor from scalar
print("tensor(3): ", x) # tensor(3)

# Create a tensor from a range
x = torch.arange(5) # 1D tensor from range
print("arange(5): ", x) # tensor([0, 1, 2, 3, 4])

x = torch.arange(1, 5) # 1D tensor from range
print("arange(1, 5): ", x) # tensor([1, 2, 3, 4])

x = torch.arange(1, 5, 0.5) # 1D tensor from range
print("arange(1, 5, 0.5): ", x) # tensor([1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000])

# Create a tensor from a linspace
x = torch.linspace(1, 5, 9) # 1D tensor from linspace
print("linspace(1, 5, 9): ", x) # tensor([1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000, 5.0000])

# Create a tensor from a logspace
x = torch.logspace(0, 2, 5) # 1D tensor from logspace
print("logspace(0, 2, 5): ", x) # tensor([  1.0000,   3.1623,  10.0000,  31.6228, 100.0000])

# Autograd in PyTorch : Automatic Differentiation 

x = torch.rand(3, requires_grad=True) # Create a tensor with requires_grad=True
y = x + 2
print(x)
print(y)
print(f"x.grad :{y.grad_fn}") # Print the gradient function


x = torch.tensor(2.0, requires_grad=True) # Create a tensor with requires_grad=True
y = 3*x**2
y.backward() # Compute the gradient
print(f"x.grad :{x.grad}") # Print the gradient






# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Define a simple neural network
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Create the model, define the loss function and the optimizer
# model = SimpleNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Dummy training loop
# for epoch in range(10):
#     optimizer.zero_grad()
#     outputs = model(torch.randn(64, 1, 28, 28))  # Dummy input
#     loss = criterion(outputs, torch.randint(0, 10, (64,)))  # Dummy target
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')