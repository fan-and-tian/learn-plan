import numpy as np
#数组创建
a = np.array([1,2,3,4,5])
b = np.arange(0,10,2)
c = np.zeros((2,3))
print("a:",a)
print("b:",b)
print("c:\n",c)

#切片
d = np.arange(12).reshape(3,4)
print("d:\n",d)
print("第二行：",d[1])
print("第一列：",d[:,0])
print("前两行后两列：\n",d[:2,2:])

#广播
x = np.array([[1],[2],[3]])
y = np.array([[10,20,30]])
z = x + y
print("x形状:",x.shape)
print("y形状:",y.shape)
print("广播结果:",z)

#向量化 与for循环比较
import time
A = np.random.rand(1000000)
B = np.random.rand(1000000)
#向量化
start = time.time()
result_np = np.sum((A-B)**2)
np_time = time.time() - start

#for循环
start = time.time()
result_for = 0
for i in range(len(A)):
    result_for += (A[i]-B[i])**2
for_time = time.time() - start
print(f"for循环: {for_time:.4f}s")
print(f"向量化: {np_time:.6f}s")
print(f"向量化快 {for_time / np_time:.0f} 倍")