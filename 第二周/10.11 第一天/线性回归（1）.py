import torch
torch.manual_seed(42)
X = torch.randn(100,1)
y = 3*X+7+torch.randn(100,1)*0.01
print(X.shape)
print(y.shape)

w = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)

lr = 0.1
for epoch in range(100):
    pred = X*w+b
    loss = ((y-pred)**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
        w.grad.zero_()
        b.grad.zero_()
print(f"\n最终: w={w.item():.4f}, b={b.item():.4f}")
print("期望: w≈3.0, b≈7.0")
