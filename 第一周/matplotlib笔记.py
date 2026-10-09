import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['KaiTi']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False
x = np.random.rand(20)*10
y = np.random.rand(20)*10

plt.scatter(x,y,color = 'blue',label ='随机点')

line_x = np.linspace(0,10,100)
line_y = 2 * line_x + 1
plt.plot(line_x,line_y,color = 'red',label = 'y=2x+1')

plt.title('第一张图')
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.legend()
plt.show()