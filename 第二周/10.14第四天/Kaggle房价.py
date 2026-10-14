import torch
import numpy as np
import pandas as pd
from torch import nn
import os
BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, 'housing.csv'))
print('shape:',df.shape)
print(df.head())

label =df['median_house_value']
all_feature = df.drop(['median_house_value'],axis=1)

numeric_feaures = all_feature.dtypes[all_feature.dtypes != 'object'].index
all_feature[numeric_feaures] = all_feature[numeric_feaures].apply(
    lambda x : (x - x.mean())/(x.std()))
all_feature[numeric_feaures] = all_feature[numeric_feaures].fillna(0)
all_feature = pd.get_dummies(all_feature, dummy_na=True)

features = torch.tensor(all_feature.values.astype(np.float32))
labels = torch.tensor(label.values, dtype=torch.float32)
labels = torch.log(labels)

torch.manual_seed(42)
perm = torch.randperm(len(features))
features = features[perm]
labels = labels[perm]

n_test = int(len(features)*0.2)
x_test = features[-n_test:]
y_test = labels[-n_test:]
x_train = features[:-n_test]
y_train = labels[:-n_test]
print(f'train:{x_train.shape},test:{x_test.shape}')

loss = nn.MSELoss()
def get_net ():
    net = nn.Sequential(nn.Linear(x_train.shape[1],1))
    return net
def log_rmse(net, features, labels):
    preds = net(features).squeeze(-1)
    return torch.sqrt(loss(preds, labels))

def train(net,x_train,y_train,x_test,y_test,
          num_epochs,lr,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    train_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,y_train),
                                             batch_size = batch_size,shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(),lr,weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X).squeeze(-1)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, x_train, y_train).item())
        test_ls.append(log_rmse(net, x_test, y_test).item())
        if epoch % 50 == 0:
            print(f"epoch {epoch}: train log_rmse={train_ls[-1]:.4f}, 验证 log_rmse={test_ls[-1]:.4f}")
    return train_ls, test_ls
net = get_net()
print(f"\n模型参数量: {sum(p.numel() for p in net.parameters()):,}")
train(net, x_train, y_train, x_test, y_test,
      num_epochs=200, lr=5e-3, weight_decay=1e-4, batch_size=64)
final_rmse = log_rmse(net, x_test, y_test).item()
print(f"\n最终验证 log_rmse: {final_rmse:.4f}")
print(f"验收标准: < 0.35")
net.eval()
with torch.no_grad():
    test_preds = net(x_test).squeeze(-1).numpy()
result = pd.DataFrame({
    '真实房价': np.exp(y_test.numpy()),
    '预测房价': np.exp(test_preds),
})
result['误差'] = result['预测房价'] - result['真实房价']
result.to_csv('预测结果.csv', index=False, encoding='utf-8-sig')
print(f"\n预测结果.csv 已生成，平均误差 ${result['误差'].abs().mean():,.0f}")


        
