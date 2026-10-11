import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
#构建数据
torch.manual_seed(42)
X = torch.randn(100,1)
y = 3*X+7+torch.randn(100,1)*0.01
#数据打包
dataset = TensorDataset(X,y)
loader = DataLoader(dataset,batch_size=16,shuffle=True)
#每次取16个 shuffle=True：打乱顺序
#模型
model = nn.Linear(in_features=1,out_features=1)
#内部自动建立了w（1x1） b（1） 且开启了自动求导
#损失函数
loss_fn = nn.MSELoss()#均方误差损失
optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)
#torch.optim.SGD:pytorch 内置 SGD 优化器不用手动写p -= lr*p.grad`和`zero_()
#model.parameters():自动收集 model 里所有可学习参数(w 和 b)
for epoch in range(100):
    for X_batch,y_batch in loader:
        pred = model(X_batch)
        loss = loss_fn(pred,y_batch)
        optimizer.zero_grad()#梯度清零
        loss.backward()
        optimizer.step()#更新参数
print(f'w={model.weight.item():.4f}')
print(f"b={model.bias.item():.4f}")
import matplotlib.pyplot as plt
plt.scatter(X.numpy(), y.numpy())
x_line = torch.linspace(X.min(), X.max(),100).reshape(-1,1)
y_line = model(x_line)
plt.plot(x_line.numpy(), y_line.detach().numpy(), 'r')
plt.show()
