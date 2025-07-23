import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置平面方程 2x + 3y + z - 6 = 0
A, B, C, D = 2, 3, 1, -6
normal_vector = np.array([A, B, C])

# 创建三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制平面
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = (-A * X - B * Y - D) / C  # 由平面方程解出 z

ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', label='Plane: 2x + 3y + z = 6')

# 固定点 P0 (选择一个平面上的点，例如 x=1, y=1 时，z=1)
x0, y0, z0 = 1, 1, 1
ax.scatter([x0], [y0], [z0], color='red', s=100, label='Fixed Point P0(1,1,1)')

# 平面上的另一点 P (例如 x=2, y=0, z=2)
x, y, z = 2, 0, 2
ax.scatter([x], [y], [z], color='blue', s=100, label='Point P(2,0,2)')

# 绘制法向量 (从原点开始，为了可视化调整长度)
ax.quiver(0, 0, 0, A, B, C, color='green', label='Normal Vector (2,3,1)', linewidth=2)

# 绘制从 P0 到 P 的方向向量
direction_vector = np.array([x - x0, y - y0, z - z0])
ax.quiver(x0, y0, z0, direction_vector[0], direction_vector[1], direction_vector[2],
          color='purple', label='Direction Vector P0->P', linewidth=2)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置图形标题
plt.title('Plane and Vectors: 2x + 3y + z = 6')

# 添加图例
ax.legend()

# 显示图形
plt.show()

# 验证点积
dot_product = np.dot(normal_vector, direction_vector)
print(f"Dot product of normal vector and direction vector: {dot_product}")
