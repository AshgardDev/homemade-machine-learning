import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# 创建 3D 图形

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# 1. 绘制平面 z = x + y
# 定义 x 和 y 的范围
x = np.arange(-1, 4, 0.1)  # 缩小范围以更好地显示圆柱体
y = np.arange(-1, 4, 0.1)
X, Y = np.meshgrid(x, y)
Z = X + Y  # 计算 z = x + y

# 绘制平面
# ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)  # alpha 增加透明度
plt.contour(x, y, Z)


# 显示图形
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 数据点
X_class1 = np.array([[1, 1], [2, 2]])
X_class_neg1 = np.array([[0, 1], [1, 0]])

# 决策超平面参数 (从上面计算得到)
# w = (2, 2), b = -3
# 超平面方程: 2*x1 + 2*x2 - 3 = 0  => x2 = -x1 + 1.5

# 绘制数据点
plt.figure(figsize=(8, 6))
plt.scatter(X_class1[:, 0], X_class1[:, 1], color='blue', label='Class +1', marker='o', s=100)
plt.scatter(X_class_neg1[:, 0], X_class_neg1[:, 1], color='red', label='Class -1', marker='x', s=100)

# 绘制决策超平面
x_vals = np.linspace(-0.5, 3, 100)
y_decision = -x_vals + 1.5
plt.plot(x_vals, y_decision, 'k-', label='Decision Hyperplane ($2x_1 + 2x_2 - 3 = 0$)', linewidth=2)

# 绘制间隔边界 (支持向量所在的位置)
# 上边界: 2*x1 + 2*x2 - 3 = 1  => x2 = -x1 + 2
y_upper_margin = -x_vals + 2
plt.plot(x_vals, y_upper_margin, 'k--', label='Margin Boundary ($2x_1 + 2x_2 - 4 = 0$)', linewidth=1)

# 下边界: 2*x1 + 2*x2 - 3 = -1 => x2 = -x1 + 1
y_lower_margin = -x_vals + 1
plt.plot(x_vals, y_lower_margin, 'k--', label='Margin Boundary ($2x_1 + 2x_2 - 2 = 0$)', linewidth=1)

# 标记支持向量
# 支持向量: (1,1), (0,1), (1,0)
plt.scatter([1, 0, 1], [1, 1, 0], s=200, facecolors='none', edgecolors='green', linewidths=2, label='Support Vectors')

plt.title('SVM Max Margin Example (Non-Axis Parallel)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([-0.5, 3])
plt.ylim([-0.5, 3])
plt.axvline(0, color='gray', linestyle=':', linewidth=0.5)
plt.axhline(0, color='gray', linestyle=':', linewidth=0.5)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()