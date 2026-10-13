import torch
import torch.nn as nn#神经网络模块包
import torch.optim as optim#内置优化器
from torchvision import datasets ,transforms
#内置数据集 数据转换工具
transform = transforms.ToTensor()
#把图片原始像素0~255转为浮点张量0~1；(归一化)
#同时把图片维度`(H,W,C)`转换成`(C,H,W)
train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data',train =True,download =True,
                                                                 transform=transform)
                                                                ,batch_size=64,shuffle=True)
#datasets.FashionMNIST()实例化数据集对象 （训练更新W，b）
#./data 存放数据集文件
#train = True 读取数据集
#download = True 无文件自动下载
#torch.utils.data.DataLoader 迭代器
test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data',train = False,transform=transform),batch_size = 1000)
model = nn.Linear(784,10)
#内部自动生成权重矩阵W:[784,10] 偏置 b:[10] 并且requires_grad=True开启梯度记录
criterion = nn.CrossEntropyLoss()#交叉熵损失函数内部包含softmax计算
optimizer = optim.SGD(model.parameters(),lr=0.01)
#model.parameters()自动抓取model里W和b
for epoch in range(20):
    total_loss = 0
    for X_batch,y_batch in train_loader:
        X_batch = X_batch.reshape(-1,784)#展平成2维供给nn.Linear
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits,y_batch)
        loss.backward()
        #pytorch自动微分核心 有梯度之后优化器才能更新参数
        optimizer.step()
        total_loss += loss.item()
    correct = 0
    with torch.no_grad():#测试阶段不需要更新梯度
        for X_batch,y_batch in test_loader:
            X_batch = X_batch.reshape(-1,784)
            preds = model(X_batch).argmax(dim=1)
            #对每行找最大得分的下标 作为预测类别
            correct += (preds == y_batch).sum().item()
        acc = correct / len(test_loader.dataset)
    if epoch % 5 == 0:
        print(f"epoch {epoch}: loss={total_loss:.4f}, 测试准确率={acc:.4f}")

print(f"\n最终测试准确率: {acc:.4f}")
print("验收标准: ≥ 0.83")