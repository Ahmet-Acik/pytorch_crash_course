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
