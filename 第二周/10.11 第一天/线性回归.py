import torch 
import matplotlib.pyplot as plt #导入画图库
plt.rcParams['font.sans-serif']= ['SimHei'] #黑体定义
plt.rcParams['axes.unicode_minus'] = False #解决负号
#1.构造数据
torch.manual_seed(42)
n = 1000 #样本数
X = torch.randn(n,2)#随机n行2个特征值
True_w = torch.tensor([[2.0],[-3.4]])
True_b = 4.2
y = X @ True_w + True_b + torch.randn(n,1)*0.01 #关系式子 torch.randn(n,1)*0.01高斯噪音
#2.小批量数据迭代器:不用一次性拿全部 1000 样本算梯度 每次只用 32 个 降低内存消耗
def load_array(batch_size):
    indices = torch.randperm(len(X))#随机下标
    for i in range(0,len(indices),batch_size):
        batch = indices[i:i+batch_size]
        yield X[batch],y[batch]
batch_size = 32
data_iter = load_array(batch_size)
#3.定义模型
w = torch.zeros(2,1,requires_grad = True)#初始化权重和偏置
b = torch.zeros(1,requires_grad = True)
def model(X):
    return X @ w + b
#4.损失函数
def squared_loss(y_pred,y_true):
    return (y_pred - y_true)**2 / 2
#5.手写SGD（随机梯度下降）优化器
def sgd(params,lr):
    with torch.no_grad():#不做画图记录
        for p in params:
            p -=lr * p.grad#梯度下降核心更新式
            p.grad.zero_()#清零梯度
#6.训练主循环
lr = 0.3#学习率
num_epochs = 5
for epoch in range(num_epochs):
    for X_batch,y_batch in load_array(batch_size):
        loss = squared_loss(model(X_batch),y_batch).mean()
        loss.backward()#反向求偏导
        sgd([w,b],lr)
    epoch_loss = squared_loss(model(X),y).mean()#mean求平均
    print(f"epoch{epoch+1},loss = {epoch_loss.item():.6f}")
    #item()用在只有一个数字的张量上 将张量包裹的数字变为浮点数
print(f"学到的 w: {w.detach().flatten().tolist()}")
print(f"学到的 b: {b.item():.4f}")
print(f"真实的 w: {True_w.tolist()}")
print(f"真实的 b: {True_b}")
print("X[:,0] 形状:", X[:, 0].shape)
print("y 形状:", y.shape)
plt.scatter(X[:, 0].numpy(), y.squeeze().numpy(), alpha=0.3, label='数据')
#取全部样本的第 0 号特征 
#.numpy()：把 PyTorch 张量 → 转成普通 numpy 数组
#alpha=0.3：透明度 0.3
x_line = torch.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_line = x_line * w[0] + b  # 固定x2=0看截面
plt.plot(x_line.numpy(), y_line.detach().numpy(), 'r', label='学到的线')
#.detach()：把这个张量从计算图剥离出来，不再关联梯度
plt.legend()
plt.title("线性回归拟合结果")
plt.show()
#笔记：广播规则——形状不同时 较小的一边会沿着缺失的维度复制 
# (1000,) 和 (1000,1) 相加时 (1000,) 被当成 (1,1000) 再复制 1000 行 
# 就爆成了 (1000,1000) 做矩阵运算前先 print(shape)检查