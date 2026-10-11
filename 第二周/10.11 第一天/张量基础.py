import torch
#建立张量
a = torch.tensor([1,2,3])
b = torch.zeros(3,3)
c = torch.randn(4,3)
print(f"a:{a}")
print(f"b:{b}")
print(f"c:{c}")

import numpy as np
np_arr = np.array([1,2,3])
t = torch.from_numpy(np_arr)
back = t.numpy()
print(f"互转成功：{back}")