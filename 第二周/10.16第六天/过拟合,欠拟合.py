import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets , transforms
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(
                './data',train=True,download=True,transform=transform),
                batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST(
                './data',train=False,download=True,transform=transform),
                batch_size=1000)

def make_model(use_dropout = False):
    layout = [nn.Linear(784,512),nn.ReLU()]
    if use_dropout:
        layout.append(nn.Dropout(0.5))
    layout += [nn.Linear(512,256),nn.ReLU(),nn.Linear(256,10)]
    return nn.Sequential(*layout)

def train_one(use_dropout=False,weight_decay = 0.0,epochs = 50,label=""):
    torch.manual_seed(42)
    model = make_model(use_dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1,weight_decay=weight_decay)
    train_losses,train_accs,val_accs=[],[],[]
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X,y in train_loader:
            X = X.reshape(-1,784)
            loss = criterion(model(X),y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        train_losses.append(total_loss)
        model.eval()
        with torch.no_grad():
            correct =0
            for i,(X,y) in enumerate(train_loader):
                if i >= 30 :
                    break
                X = X.reshape(-1,784)
                correct += (model(X).argmax(dim =1)==y).sum().item()
            train_acc = correct/2000
        correct = 0
        for X,y in test_loader:
            X = X.reshape(-1,784)
            correct += (model(X).argmax(1) == y).sum().item()
        val_acc = correct / len(test_loader.dataset)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if epoch % 5 == 0:
            print(f"[{label}] epoch {epoch}: loss={total_loss:.1f}, "
                f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    return  train_losses, train_accs, val_accs
print("===== 实验1: Baseline =====")
loss1, tr1, va1 = train_one(label="baseline")

print("\n===== 实验2: 加 Dropout(0.5) =====")
loss2, tr2, va2 = train_one(use_dropout=True, label="dropout")

print("\n===== 实验3: 加权重衰减 weight_decay=0.01 =====")
loss3, tr3, va3 = train_one(weight_decay=0.01, label="wd")

# ===== 结果对比表 =====
print("\n===== 最终对比 =====")
print(f"{'配置':<20} {'训练acc':>8} {'验证acc':>8} {'差距':>8}")
print(f"{'baseline':<20} {tr1[-1]:>8.4f} {va1[-1]:>8.4f} {tr1[-1]-va1[-1]:>8.4f}")
print(f"{'dropout':<20} {tr2[-1]:>8.4f} {va2[-1]:>8.4f} {tr2[-1]-va2[-1]:>8.4f}")
print(f"{'weight_decay':<20} {tr3[-1]:>8.4f} {va3[-1]:>8.4f} {tr3[-1]-va3[-1]:>8.4f}")

#画三组的验证集准确率曲线 
plt.figure(figsize=(10, 6))
plt.plot(va1, label='baseline')
plt.plot(va2, label='Dropout(0.5)')
plt.plot(va3, label='weight_decay=0.01')
plt.xlabel('epoch')
plt.ylabel('验证集准确率')
plt.title('防过拟合三组对比')
plt.legend()
plt.grid(True)
plt.savefig('防过拟合对比.png', dpi=120)
plt.show()

