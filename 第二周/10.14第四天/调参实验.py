import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE,'housing.csv'))

label = df['median_house_value']
all_feature = df.drop(['median_house_value'],axis=1)

numeric_feature = all_feature.dtypes[all_feature.dtypes != 'object'].index
all_feature = all_feature[numeric_feature].apply(
    lambda x: (x - x.mean())/(x.std()))
all_feature = all_feature[numeric_feature].fillna(0)
all_feature = pd.get_dummies(all_feature,dummy_na=True)

features = torch.tensor(all_feature.values.astype(np.float32))
labels = torch.tensor(label.values,dtype=torch.float32)
labels = torch.log(labels)

torch.manual_seed(42)
perm = torch.randperm(len(features))
features = features[perm]
labels = labels[perm]
n_test = int(len(features)*0.2)
X_test = features[-n_test:]
y_test = labels[-n_test:]
X_train = features[:-n_test]
y_train = labels[:-n_test]

loss = nn.MSELoss()
def get_net(hidden1 = 0,hidden2 = 0,use_dropout = False):
    if hidden1 == 0:
        return nn.Sequential(nn.Linear(X_train.shape[1],1))
    layers = [nn.Linear(X_train.shape[1],hidden1),nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(hidden1,hidden2))
    layers.append(nn.ReLU())
    if use_dropout:
        layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(hidden2,1))
    return nn.Sequential(*layers)
def log_rmse(net,features,labels):
    net.eval()
    with torch.no_grad():
        preds = net(features).squeeze(-1)
        return torch.sqrt(loss(preds,labels))
def train(net, X_train, y_train, X_test, y_test,
          num_epochs, lr, weight_decay, batch_size):
    train_ls,test_ls = [],[]
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train,y_train),
        batch_size=batch_size,shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr
                                 ,weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X).squeeze(-1)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,X_train,y_train).item())
        test_ls.append(log_rmse(net,X_test,y_test).item())
    return train_ls,test_ls

configs = [
   # (名称, hidden1, hidden2, use_dropout, lr, weight_decay, batch_size)
    ("baseline 线性回归",      0,   0,  False, 5e-3, 1e-4, 64),
    ("lr=1e-2",              0,   0,  False, 1e-2, 1e-4, 64),
    ("加隐藏层 128/64",     128,  64,  False, 5e-3, 1e-4, 64),
    ("加隐藏层+Dropout",    128,  64,  True,  5e-3, 1e-4, 64),
]
result = []
for name , h1,h2,drop,lr,wd,bs in configs:
    torch.manual_seed(42)
    net = get_net(h1,h2,drop)
    param_count = sum(p.numel() for p in net.parameters())
    train(net, X_train, y_train, X_test, y_test,
          num_epochs=200, lr=lr, weight_decay=wd, batch_size=bs)
    final_rmse = log_rmse(net,X_test,y_test).item()
    result.append((name,param_count,lr,wd,bs,final_rmse))
    print(f'名称:{name},参数量:{param_count},:log_rmse:{final_rmse}')


print("\n" + "=" * 70)
print(f"{'配置':<22} {'参数量':>8} {'lr':>8} {'weight_decay':>12} {'batch':>6} {'log_rmse':>9}")
print("-" * 70)
for name, pc, lr, wd, bs, rmse in result:
    print(f"{name:<22} {pc:>8,} {lr:>8.0e} {wd:>12.0e} {bs:>6} {rmse:>9.4f}")
print("-" * 70)
best = min(result, key=lambda x: x[-1])
print(f"\n最优配置: {best[0]}, log_rmse = {best[-1]:.4f}")
print(f"和 baseline 的差距: {result[0][-1] - best[-1]:+.4f}")
