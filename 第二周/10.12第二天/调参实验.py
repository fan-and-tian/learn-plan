import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms

def train_and_evaluate(lr,batch_size,epochs):
    transform = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data',train = True,download=True,transform=transform)
        ,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST
                                              ('./data',train=False,download=True,transform=transform)
                                              ,batch_size=1000)
    torch.manual_seed(42)
    model = nn.Linear(784,10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=lr)
    for epoch in range(epochs):
        for X_batch,y_batch in train_loader:
            X_batch = X_batch.reshape(-1,784)
            optimizer.zero_grad()
            loss = criterion(model(X_batch),y_batch)
            loss.backward()
            optimizer.step()
            correct = 0
        with torch.no_grad():
            for X_batch,y_batch in test_loader:
                X_batch =X_batch.reshape(-1,784)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds==y_batch).sum().item()
        return correct / len(test_loader.dataset)

# 三组对比实验
print("实验1: 学习率对比")
print(f"  lr=0.01, epoch=10: {train_and_evaluate(0.01, 64, 10):.4f}")
print(f"  lr=0.1,  epoch=10: {train_and_evaluate(0.1, 64, 10):.4f}")
print("实验2: batch size对比")
print(f"  batch=32,  lr=0.1: {train_and_evaluate(0.1, 32, 10):.4f}")
print(f"  batch=128, lr=0.1: {train_and_evaluate(0.1, 128, 10):.4f}")
print("实验3: 训练轮数对比")
print(f"  epoch=5,  lr=0.1: {train_and_evaluate(0.1, 64, 5):.4f}")
print(f"  epoch=20, lr=0.1: {train_and_evaluate(0.1, 64, 20):.4f}")
