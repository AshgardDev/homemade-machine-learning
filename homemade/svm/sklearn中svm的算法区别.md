是的，`sklearn` 的 `svm` 模块在某些模型中 **确实使用了对偶形式（dual form）进行训练和推理**，特别是以下这几种情况：

---

## ✅ 会使用对偶形式的模型

### 1. `sklearn.svm.SVC`（支持核函数）

```python
from sklearn.svm import SVC
clf = SVC(kernel='rbf')  # 默认就是 RBF 核
```

* ✅ **使用了对偶形式**
* ✅ 利用了拉格朗日乘子 $\alpha_i$
* ✅ 使用核函数 $K(x_i, x_j)$，避免显式映射高维空间

即，决策函数是：

$$
f(x) = \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b
$$

其中 SV 是支持向量集合，`clf.dual_coef_` 就是 $\alpha_i y_i$，你可以直接访问。

---

### 2. `sklearn.svm.NuSVC`（基于对偶的另一种参数化）

```python
from sklearn.svm import NuSVC
clf = NuSVC(kernel='poly')
```

* ✅ 同样使用对偶形式
* ✅ 使用核技巧进行非线性分类
* ✅ 支持访问 `dual_coef_` 等属性

---

## ❌ 不使用对偶形式的模型

### 3. `sklearn.svm.LinearSVC`

```python
from sklearn.svm import LinearSVC
clf = LinearSVC()
```

* ❌ **不使用对偶形式**
* ✅ 使用 **原始问题（primal form）** 做优化
* ✅ 优化目标为：

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i} \max(0, 1 - y_i (\mathbf{w}^\top x_i + b))
$$

* 优化方法通常是 **liblinear**，基于坐标下降或拟牛顿法，不用核函数

---

## 🔍 总结：哪些用对偶形式？

| 模型          | 使用对偶 | 是否支持核函数        | 优化方式           |
| ----------- | ---- | -------------- | -------------- |
| `SVC`       | ✅ 是  | ✅ 是（支持 kernel） | 对偶，SMO         |
| `NuSVC`     | ✅ 是  | ✅ 是            | 对偶，SMO         |
| `LinearSVC` | ❌ 否  | ❌ 否            | 原始形式，liblinear |

---

## 📌 怎么看是否用了对偶形式？

```python
from sklearn.svm import SVC
clf = SVC(kernel='rbf').fit(X, y)

# 对偶变量：
print(clf.dual_coef_)   # (1, n_support_vectors)
print(clf.support_vectors_)  # (n_support_vectors, n_features)
```

如果模型中能访问 `dual_coef_` 和 `support_vectors_`，说明它使用了对偶形式。

---

需要我展示下 SVC 的决策函数是怎么组合核函数和对偶变量的吗？还是你想看看 dual 与 primal 解的比较？
