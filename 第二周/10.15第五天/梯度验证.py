import torch
def f(x):
    return(x**2+3*x).sum()

x = torch.tensor([2.0,3.0,4.0],requires_grad=True)
y = f(x)
y.backward()
print(x.grad) 
x2 = torch.tensor([2.0, 3.0, 4.0])
h = 1e-4
grad_numeric = (f(x2 + h) - f(x2 - h)) / (2 * h)
print("数值微分梯度:", grad_numeric)
print("误差:", (x.grad - grad_numeric).abs().max().item())