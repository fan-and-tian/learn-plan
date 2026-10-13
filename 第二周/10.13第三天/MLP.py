import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils import data
batch_size = 256
torch.manual_seed(42)
def load_data_fashion_mnist(batch_size):
    transform = transforms.ToTensor()
    mnist_train = datasets.FashionMNIST('./data',train= True
                                        ,download=True,transform=transform)
    mnist_test = datasets.FashionMNIST('./data',train=False,
                                       download=True,transform=transform)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True),
            data.DataLoader(mnist_test,batch_size))

train_iter , test_iter = load_data_fashion_mnist(batch_size)
num_inputs , num_outputs, num_hiddens = 784,10,256
W1 = nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01) 
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params = [W1,b1,W2,b2]

def relu(Z):
    Z = Z.clone()
    Z[Z<0] = 0
    return Z

def net(X):
    X = X.reshape(-1,num_inputs)
    H = relu(X@W1+b1)
    return H@W2+b2

loss = nn.CrossEntropyLoss()

num_epoch , lr = 10,0.1
updater = torch.optim.SGD(params,lr=lr)

for epoch in range(num_epoch):
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X ,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        l.backward()
        updater.step()
        updater.zero_grad()
        train_loss_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
        n += y.numel()
    print(f"epoch ={epoch + 1}, "
          f"loss ={train_loss_sum / n:.4f}, "
          f"acc ={train_acc_sum / n:.4f}")
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for X ,y in test_iter:
            acc_sum += (net(X).argmax(dim =1)==y).sum().item()
            n += y.numel()
        print(f"test acc={acc_sum/n:.4}")
        
