import torch
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,10)
)
print(model)
for p in model.parameters():
    print(f'shape{tuple(p.shape)},require grad:{p.requires_grad}')

x = torch.rand(64,784)
output = model(x)
print(f'shape:{x.shape}')
print(f'shape:{output.shape}')

def init_weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

model.apply(init_weights)
print(f"第一层weight范围: [{model[0].weight.min():.4f}, {model[0].weight.max():.4f}]")

first_layer_weight = model[0].bias
print(f"\n第一层weight形状: {first_layer_weight.shape}")   
