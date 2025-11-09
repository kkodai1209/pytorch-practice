import torch

x = torch.tensor(3.0, requires_grad = True)

y = x*x

y.backward()

print(f"x = {x}")
print(f"y = x*x ={y}")
print(f"Gradient (dy/dx) at x = 3.0 is:{x.grad}")