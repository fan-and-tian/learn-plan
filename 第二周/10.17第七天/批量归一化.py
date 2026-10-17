import torch
import torch.nn as nn
from torchvision import datasets , transforms
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('./data',download=True,train=True,transform=transform)
                ,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('./data',download=True,train=False,transform=transform)
                ,batch_size=1000)
def make_mode(use_BN = False):
    layout = [nn.Linear(784,512)]
    if use_BN:
        layout.append(nn.BatchNorm1d(512))
    layout += [nn.ReLU(),nn.Linear(512,256)]
    if use_BN:
        layout += [nn.BatchNorm1d(256)]
    layout += [nn.ReLU(),nn.Linear(256,10)]
    return nn.Sequential(*layout)
def train_one(use_BN,epochs,label):
    torch.manual_seed(42)
    model = make_mode(use_BN)
    optimimzer = optim.SGD(model.parameters(),lr = 0.1)
    loss = nn.CrossEntropyLoss()
    train_losses ,val_accs , epoch_to_85 = [],[],None
    for epoch in range(epochs+1):
        model.train()
        total_losses = 0
        for X,y in train_loader:
            X = X.reshape(-1,784)
            optimimzer.zero_grad()
            l = loss(model(X),y)
            l.backward()
            optimimzer.step()
            total_losses += l.item()
        train_losses.append(total_losses)
        model.eval()
        with torch.no_grad():
            correct = 0
            for X,y in test_loader:
                X = X.reshape(-1,784)
                correct += (model(X).argmax(dim=1)==y).sum().item()
            val_acc = correct/len(test_loader.dataset)
        val_accs.append(val_acc)
        if epoch_to_85 == None and val_acc>=0.85:
            epoch_to_85 = epoch + 1
        if epoch % 5 == 0:
            print(f"[{label}] epoch {epoch}: loss={total_losses:.1f}, val_acc={val_acc:.4f}")
    return train_losses, val_accs, epoch_to_85

print("===== 无 BatchNorm =====")
loss1, acc1, e1 = train_one(use_BN=False, epochs=20, label="无BN")
print("\n===== 有 BatchNorm =====")
loss2, acc2, e2 = train_one(use_BN=True, epochs=20, label="有BN")
print(f"{'配置':<12} {'最终val_acc':>10} {'达85%用时':>10}")
print(f"{'无BN':<12} {acc1[-1]:>10.4f} {str(e1):>10}")
print(f"{'有BN':<12} {acc2[-1]:>10.4f} {str(e2):>10}")

def count_params(model):
    return sum(p.numel()for p in model.parameters())
model_no_bn = make_mode(use_BN=False)
model_bn = make_mode(use_BN=True)
print(f'有BN参数量{count_params(model_bn)}')
print(f'无BN参数量{count_params(model_no_bn)}')

fig , (ax1,ax2) = plt.subplots(1,2,figsize = (14,5))
ax1.plot(loss1,label = '无BN')
ax1.plot(loss2,label = '有BN')
ax1.set_xlabel('epoch')
ax1.set_ylabel('训练loss')
ax1.set_title('训练loss对比')
ax1.legend()
ax1.grid(True)
ax2.plot(acc1, label='无BN')
ax2.plot(acc2, label='有BN')
ax2.set_xlabel('epoch')
ax2.set_ylabel('验证准确率')
ax2.set_title('验证准确率对比')
ax2.legend()
ax2.grid(True)
plt.savefig('BatchNorm对比.png', dpi=120)
plt.show()
#BatchNorm 的主要收益是收敛速度（达 85% 快 4 倍）和早期稳定性（消除前几个 epoch 的震荡），
# 对最终准确率提升有限（+0.25 个点）
#它的核心价值是让深网络能训、训得稳，而不是训得更高。