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


import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create the model, define the loss function and the optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(torch.randn(64, 1, 28, 28))  # Dummy input
    loss = criterion(outputs, torch.randint(0, 10, (64,)))  # Dummy target
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')