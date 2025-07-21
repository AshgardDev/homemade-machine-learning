随机森林（Random Forest）和 Bagging 是机器学习中两种重要的集成学习方法，它们在思想上有密切的关系，但也有显著的区别。结合您之前的上下文（Spring Boot、Python、机器学习如 KNN、决策树、Iris 数据集等），我将简要分析它们的**关系**和**区别**，并提供与 Java 和 Python 的相关示例，保持简洁且实用。

### 1. **关系**
- **Bagging 是随机森林的基础**：
  - Bagging（Bootstrap Aggregating）是一种集成学习方法，通过从原始数据集有放回地抽样（bootstraprestrictive sampling）生成多个子数据集，训练多个模型（通常决策树），并通过投票（分类）或平均（回归）聚合预测结果。
  - 随机森林是 Bagging 的一种扩展，结合了 Bagging 的思想并引入了额外的随机性（特征随机选择）。

- **共同点**：
  - 两者都基于**决策树**作为基学习器。
  - 都使用 **Bootstrap Sampling**（有放回抽样）生成多个训练子集。
  - 都通过集成多个模型的预测结果（投票或平均）提高模型的鲁棒性和准确性。
  - 都属于**集成学习**，旨在减少过拟合和提高泛化能力。

### 2. **区别**
| **特性**                | **Bagging**                                     | **随机森林**                                    |
|-------------------------|------------------------------------------------|------------------------------------------------|
| **定义**               | 通用集成方法，使用 Bootstrap 抽样训练多个独立模型并聚合结果。 | Bagging 的特化形式，增加了特征随机选择。         |
| **特征选择**           | 每次训练使用所有特征。                          | 每次节点分裂时随机选择部分特征（feature bagging）。 |
| **随机性**             | 仅通过样本抽样引入随机性。                      | 通过样本抽样和特征随机选择引入更多随机性。       |
| **模型相关性**         | 基模型（决策树）间相关性较高，可能导致性能提升有限。 | 特征随机选择降低模型相关性，提升集成效果。       |
| **性能**              | 通常比单一决策树好，但可能不如随机森林。         | 通常优于普通 Bagging，泛化能力更强。            |
| **复杂度**             | 实现较简单，计算开销较低。                      | 因特征随机选择，计算复杂度略高。                |

- **核心区别**：
  - 随机森林在 Bagging 的基础上，通过在每个决策树节点分裂时随机选择一部分特征（而非全部特征）来构建树，从而降低树与树之间的相关性，提高模型的多样性和性能。
  - Bagging 使用所有特征，模型间相关性较高，可能限制性能提升。

### 3. **使用场景**
- **Bagging**：
  - 适用于任何基学习器（如决策树、SVM），当希望通过集成减少方差时。
  - 适合简单模型或数据特征较少的情况。
- **随机森林**：
  - 专为决策树设计，适合高维数据（如特征较多时）。
  - 广泛用于分类（如 Iris 数据集分类）、回归、特征重要性评估。

### 4. **总结**
- **关系**：
  - 随机森林是 Bagging 的特化形式，增加了特征随机选择以降低模型相关性。
  - 两者都使用 Bootstrap 抽样和决策树，旨在提高泛化能力。
- **区别**：
  - Bagging 使用所有特征，模型相关性较高；随机森林通过随机特征选择降低相关性，提升性能。
  - 随机森林专为决策树设计，Bagging 可用于任何基学习器。


在 scikit-learn（sklearn）中，Bagging 算法可以通过 `BaggingClassifier` 或 `BaggingRegressor` 实现，结合决策树作为基评估器（base estimator），并支持随机样本（bootstrap sampling）和随机特征（feature subsampling）。以下是关于 Bagging 算法的详细说明，以及如何在 sklearn 中使用决策树作为基评估器实现随机样本和随机特征的配置。

### 1. **Bagging 算法简介**
Bagging（Bootstrap Aggregating）是一种集成学习方法，通过以下步骤减少模型的方差，提高泛化能力：
- **随机样本**：通过有放回抽样（bootstrap）从训练数据中生成多个子样本。
- **随机特征**：在每次构建基评估器（如决策树）时，随机选择一部分特征进行训练。
- **聚合**：将多个基评估器的预测结果进行聚合（如分类问题取多数投票，回归问题取平均值）。

当基评估器是决策树时，Bagging 算法类似于随机森林（Random Forest）的核心思想，但随机森林在特征选择上更进一步（每次分裂随机选择特征子集），而 Bagging 默认在构建每棵树时随机选择特征子集。

### 2. **sklearn 中实现 Bagging 算法**
sklearn 提供了 `BaggingClassifier`（分类）和 `BaggingRegressor`（回归）类，支持自定义基评估器（如决策树）以及随机样本和随机特征的设置。

#### 关键参数
- `base_estimator`（或 `estimator`）：基评估器，例如 `DecisionTreeClassifier` 或 `DecisionTreeRegressor`。
- `n_estimators`：基评估器的数量（即生成多少棵决策树）。
- `max_samples`：每个基评估器使用的样本比例或数量（控制随机样本）。
- `bootstrap`：是否使用有放回抽样（默认 `True`）。
- `max_features`：每个基评估器使用的特征比例或数量（控制随机特征）。
- `bootstrap_features`：是否对特征进行有放回抽样（默认 `False`）。

#### 示例代码
以下是一个使用 `BaggingClassifier` 的例子，基评估器为决策树，支持随机样本和随机特征：

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 配置 BaggingClassifier
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),  # 基评估器为决策树
    n_estimators=10,                         # 使用 10 棵决策树
    max_samples=0.8,                         # 每棵树使用 80% 的样本（随机抽样）
    max_features=0.8,                        # 每棵树使用 80% 的特征（随机特征）
    bootstrap=True,                          # 样本有放回抽样
    bootstrap_features=False,                # 特征无放回抽样
    random_state=42
)

# 训练模型
bagging.fit(X_train, y_train)

# 预测
y_pred = bagging.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

#### 代码说明
- **数据集**：使用 `make_classification` 生成一个包含 1000 个样本、20 个特征的分类数据集。
- **随机样本**：`max_samples=0.8` 表示每棵决策树使用 80% 的训练样本（通过 bootstrap 有放回抽样）。
- **随机特征**：`max_features=0.8` 表示每棵决策树使用 80% 的特征（随机选择）。
- **决策树**：`DecisionTreeClassifier` 作为基评估器，默认不限制树深度，类似于随机森林中的单棵树。
- **评估**：使用 `accuracy_score` 计算模型在测试集上的准确率。

### 3. **Bagging vs. 随机森林**
- **相似点**：两者都使用决策树作为基评估器，支持随机样本和随机特征。
- **不同点**：
  - 随机森林（`RandomForestClassifier`/`RandomForestRegressor`）在每棵树的每个节点分裂时随机选择特征子集，而 Bagging 只在构建每棵树时随机选择特征子集。
  - 随机森林是专门优化的集成方法，Bagging 是一个通用的框架，可以搭配其他基评估器（如 SVM、KNN 等）。
  - 如果你设置 `BaggingClassifier` 的 `base_estimator=DecisionTreeClassifier`、`max_samples=1.0`、`max_features=1.0`、`bootstrap=True`，效果接近随机森林，但仍因特征选择方式不同而有所差异。

#### 转换为随机森林的效果
如果你希望 Bagging 的行为更接近随机森林，可以在决策树中限制每次分裂的特征选择：

```python
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_features="sqrt"),  # 每次分裂随机选择 sqrt(n_features) 个特征
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=False,
    random_state=42
)
```

这里 `max_features="sqrt"` 模拟了随机森林在节点分裂时的随机特征选择。

### 4. **注意事项**
- **随机样本比例**：`max_samples` 过小可能导致每棵树训练数据不足，模型欠拟合；过大则可能减少多样性。
- **随机特征比例**：`max_features` 过小会降低单棵树的性能，过大则可能增加模型相关性，降低集成效果。
- **计算复杂度**：Bagging 的计算开销与 `n_estimators` 和基评估器的复杂度成正比，决策树通常较快，但特征和样本量较大时需注意。
- **超参数调优**：可以通过网格搜索（`GridSearchCV`）调整 `n_estimators`、`max_samples`、`max_features` 等参数。

### 5. **实际应用场景**
- **分类问题**：如垃圾邮件检测、疾病诊断等，使用 `BaggingClassifier`。
- **回归问题**：如房价预测、销量预测等，使用 `BaggingRegressor`。
- **高方差数据**：Bagging 特别适合高方差的模型（如未剪枝的决策树），通过集成降低过拟合风险。

如果你有具体的数据集或任务需求（例如需要处理不平衡数据、调整超参数等），可以进一步提供细节，我可以为你定制代码或优化建议！


下面是一个使用 NumPy 实现 `BaggingClassifier` 的示例，基评估器为决策树，支持随机样本（bootstrap sampling）和随机特征（feature subsampling）。由于 scikit-learn 的 `DecisionTreeClassifier` 是高效的实现，我们将使用 NumPy 实现 Bagging 的核心逻辑（随机采样、特征选择和投票集成），并调用 scikit-learn 的决策树作为基评估器。如果你需要完全从头实现的决策树，可以进一步说明。

### 实现思路
1. **随机样本**：通过 NumPy 的随机索引生成 bootstrap 样本（有放回抽样）。
2. **随机特征**：为每棵树随机选择一部分特征。
3. **基评估器**：使用 scikit-learn 的 `DecisionTreeClassifier` 作为基评估器，训练每棵树。
4. **集成预测**：对所有树的预测结果进行多数投票。
5. **NumPy 核心**：使用 NumPy 处理数据采样、特征选择和预测聚合。

以下是完整的 NumPy 实现的代码，包含 `BaggingClassifier` 的自定义类。

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

class CustomBaggingClassifier:
    def __init__(self, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, random_state=None):
        """
        Custom Bagging Classifier using NumPy.
        
        Parameters:
        - n_estimators: Number of base estimators (decision trees).
        - max_samples: Proportion or number of samples to draw for each base estimator.
        - max_features: Proportion or number of features to draw for each base estimator.
        - bootstrap: Whether to use bootstrap sampling (True) or not (False).
        - random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_features_ = []
        
    def fit(self, X, y):
        """
        Fit the bagging classifier.
        
        Parameters:
        - X: Training data (numpy array, shape [n_samples, n_features]).
        - y: Target labels (numpy array, shape [n_samples]).
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Convert max_samples and max_features to integers if they are proportions
        max_samples = int(self.max_samples * n_samples) if isinstance(self.max_samples, float) else self.max_samples
        max_features = int(self.max_features * n_features) if isinstance(self.max_features, float) else self.max_features
        
        self.estimators_ = []
        self.estimator_features_ = []
        
        for _ in range(self.n_estimators):
            # Create decision tree
            tree = DecisionTreeClassifier(random_state=np.random.randint(0, 10000))
            
            # Bootstrap sampling
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, size=max_samples, replace=True)
            else:
                sample_indices = np.random.choice(n_samples, size=max_samples, replace=False)
            
            # Random feature selection
            feature_indices = np.random.choice(n_features, size=max_features, replace=False)
            
            # Subset data
            X_subset = X[sample_indices][:, feature_indices]
            y_subset = y[sample_indices]
            
            # Train decision tree
            tree.fit(X_subset, y_subset)
            
            # Store estimator and selected features
            self.estimators_.append(tree)
            self.estimator_features_.append(feature_indices)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        - X: Input data (numpy array, shape [n_samples, n_features]).
        
        Returns:
        - Predicted class labels (numpy array, shape [n_samples]).
        """
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_estimators), dtype=int)
        
        # Collect predictions from each tree
        for i, (tree, feature_indices) in enumerate(zip(self.estimators_, self.estimator_features_)):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        
        # Majority voting
        y_pred, _ = mode(predictions, axis=1)
        return y_pred.flatten()

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train custom bagging classifier
    bagging = CustomBaggingClassifier(
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = bagging.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
```

### 代码说明
1. **类结构**：
   - `CustomBaggingClassifier` 实现了 Bagging 的核心功能：初始化、训练和预测。
   - 初始化参数包括 `n_estimators`（树的数量）、`max_samples`（样本比例）、`max_features`（特征比例）、`bootstrap`（是否放回抽样）和 `random_state`（随机种子）。

2. **训练过程（fit）**：
   - 使用 `np.random.choice` 实现 bootstrap 样本抽样（有放回）或无放回抽样。
   - 为每棵树随机选择特征子集（`max_features`）。
   - 使用 scikit-learn 的 `DecisionTreeClassifier` 作为基评估器，训练每棵树。
   - 保存每棵树的模型和对应的特征索引。

3. **预测过程（predict）**：
   - 对每棵树，使用其对应的特征子集进行预测。
   - 使用 `scipy.stats.mode` 实现多数投票，得到最终预测结果。

4. **NumPy 使用**：
   - 数据采样和特征选择使用 `np.random.choice`。
   - 数据子集提取使用 NumPy 的数组索引。
   - 预测结果聚合使用 NumPy 数组和 `scipy.stats.mode`。

5. **示例数据**：
   - 使用 `make_classification` 生成 1000 个样本、20 个特征的分类数据集。
   - 训练集和测试集按 7:3 划分，评估模型准确率。

### 运行结果
运行上述代码会输出模型在测试集上的准确率，类似于 scikit-learn 的 `BaggingClassifier`。由于使用了随机样本和随机特征，模型具有较低的方差和较好的泛化能力。

### 注意事项
- **依赖**：需要安装 `numpy`、`scipy` 和 `scikit-learn`（`pip install numpy scipy scikit-learn`）。
- **性能**：此实现依赖 scikit-learn 的决策树，效率较高。如果需要完全从头实现决策树，需额外实现树的构建逻辑（会显著增加代码复杂性）。
- **随机性**：通过 `random_state` 控制随机种子，确保结果可重现。
- **扩展性**：可以修改 `base_estimator` 为其他模型（如 KNN、SVM），但需调整输入格式。

如果需要更详细的解释、完全从头的决策树实现，或针对特定数据集的优化，请告诉我！

你的疑问非常好！确实，在经典的 Bagging（Bootstrap Aggregating）算法定义中，核心思想是通过 **随机样本**（bootstrap sampling）来构建多个基评估器（通常是决策树），然后通过集成（如投票或平均）减少模型的方差。然而，在实际应用中，特别是在 scikit-learn 的实现中，Bagging 算法被扩展为支持 **随机特征**（feature subsampling），这使得它在功能上更加灵活，并且与随机森林（Random Forest）有一定的相似性。以下是对这个问题的详细解答。

### 1. **经典 Bagging 的定义**
在原始的 Bagging 算法（由 Leo Breiman 在 1996 年提出）中：
- **随机样本**：通过有放回抽样（bootstrap）从训练数据中生成多个子样本，每个子样本用于训练一个独立的基评估器（通常是决策树）。
- **无随机特征**：经典 Bagging 通常使用全部特征来训练每个基评估器，不涉及特征的随机选择。
- **集成**：通过多数投票（分类）或平均（回归）聚合多个基评估器的预测结果。

因此，在理论上，经典 Bagging 只强调随机样本（`bootstrap=True`），而不涉及随机特征。

### 2. **scikit-learn 中 Bagging 的扩展**
在 scikit-learn 的 `BaggingClassifier` 和 `BaggingRegressor` 实现中，Bagging 被设计为一个更通用的集成学习框架，增加了对随机特征的支持，通过以下参数实现：
- **`max_features`**：控制每个基评估器使用的特征比例或数量（默认值为 1.0，表示使用所有特征）。
- **`bootstrap_features`**：控制是否对特征进行有放回抽样（默认值为 `False`，即无放回）。

这意味着 scikit-learn 的 Bagging 允许用户在构建每个基评估器时随机选择一部分特征（feature subsampling），这并不是经典 Bagging 的标准定义，而是对其功能的扩展。这种扩展使得 Bagging 在某些场景下更接近随机森林，尤其当基评估器是决策树时。

#### 为什么增加随机特征？
- **提高多样性**：随机选择特征可以进一步增加基评估器之间的差异性（diversity），从而降低模型的相关性，提升集成的泛化能力。
- **灵活性**：scikit-learn 的 Bagging 设计为通用框架，可以搭配任何基评估器（如决策树、SVM、KNN 等），而随机特征为这些基评估器提供了额外的随机性选项。
- **与随机森林的联系**：当基评估器是决策树，且设置了 `max_features<1.0`，Bagging 的行为类似于随机森林，但随机森林在每次节点分裂时随机选择特征，而 Bagging 在整个树构建时选择特征子集。

### 3. **Bagging vs. 随机森林的随机特征区别**
- **Bagging 的随机特征**：
  - 通过 `max_features` 参数，在构建每棵树时随机选择一部分特征（例如 80% 的特征）。
  - 选定的特征子集用于整棵树的训练，树的每次分裂都基于这部分特征。
  - 由 `bootstrap_features` 控制特征抽样是否放回（通常无放回）。
- **随机森林的随机特征**：
  - 通过 `max_features` 参数（例如 `"sqrt"` 或 `"log2"`），在决策树的 **每次节点分裂** 时随机选择一个特征子集。
  - 这种节点级别的随机特征选择使得随机森林的基评估器更加多样化，相比 Bagging 通常具有更低的模型相关性。

因此，scikit-learn 的 Bagging 支持随机特征是为了增强灵活性，但它的随机特征机制与随机森林的实现方式不同。

### 4. **澄清你的疑问**
你提到的“Bagging 不是说只随机样本吗，为什么还能随机特征”是非常合理的，因为经典 Bagging 确实只涉及随机样本（bootstrap）。然而：
- scikit-learn 的 `BaggingClassifier` 和 `BaggingRegressor` 扩展了经典 Bagging，增加了随机特征的功能（通过 `max_features` 和 `bootstrap_features`）。
- 如果你希望实现严格的经典 Bagging，只需设置 `max_features=1.0`（即使用所有特征），这样就只有随机样本（`bootstrap=True`）生效。

### 5. **代码验证经典 Bagging（仅随机样本）**
以下是修改后的 NumPy 实现的 `CustomBaggingClassifier`，严格遵循经典 Bagging，只进行随机样本抽样，不进行随机特征选择：

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

class ClassicBaggingClassifier:
    def __init__(self, n_estimators=10, max_samples=1.0, bootstrap=True, random_state=None):
        """
        Classic Bagging Classifier using NumPy (only random sampling, no feature subsampling).
        
        Parameters:
        - n_estimators: Number of base estimators (decision trees).
        - max_samples: Proportion or number of samples to draw for each base estimator.
        - bootstrap: Whether to use bootstrap sampling (True) or not (False).
        - random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []
        
    def fit(self, X, y):
        """
        Fit the bagging classifier.
        
        Parameters:
        - X: Training data (numpy array, shape [n_samples, n_features]).
        - y: Target labels (numpy array, shape [n_samples]).
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Convert max_samples to integer if it's a proportion
        max_samples = int(self.max_samples * n_samples) if isinstance(self.max_samples, float) else self.max_samples
        
        self.estimators_ = []
        
        for _ in range(self.n_estimators):
            # Create decision tree
            tree = DecisionTreeClassifier(random_state=np.random.randint(0, 10000))
            
            # Bootstrap sampling
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, size=max_samples, replace=True)
            else:
                sample_indices = np.random.choice(n_samples, size=max_samples, replace=False)
            
            # Subset data (use all features)
            X_subset = X[sample_indices]
            y_subset = y[sample_indices]
            
            # Train decision tree
            tree.fit(X_subset, y_subset)
            
            # Store estimator
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        - X: Input data (numpy array, shape [n_samples, n_features]).
        
        Returns:
        - Predicted class labels (numpy array, shape [n_samples]).
        """
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_estimators), dtype=int)
        
        # Collect predictions from each tree
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)
        
        # Majority voting
        y_pred, _ = mode(predictions, axis=1)
        return y_pred.flatten()

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train classic bagging classifier
    bagging = ClassicBaggingClassifier(
        n_estimators=10,
        max_samples=0.8,
        bootstrap=True,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = bagging.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
```

### 代码说明
- **经典 Bagging**：删除了 `max_features` 和 `bootstrap_features` 参数，确保每个基评估器使用全部特征，仅通过 `max_samples` 和 `bootstrap` 控制随机样本。
- **差异**：与之前的 `CustomBaggingClassifier` 相比，此实现不进行特征子集选择，严格遵循经典 Bagging 的定义。
- **结果**：由于没有随机特征，基评估器之间的多样性可能略低于支持随机特征的版本，但仍然通过随机样本降低方差。

### 6. **总结**
- **经典 Bagging**：只涉及随机样本（bootstrap sampling），每个基评估器使用全部特征。
- **scikit-learn 的 Bagging**：扩展了经典 Bagging，增加了随机特征的支持（通过 `max_features`），使其更灵活且接近随机森林。
- **为什么有随机特征**：scikit-learn 的设计目标是提供通用的集成框架，随机特征可以增强基评估器的多样性，改善模型性能。
- **随机森林的区别**：随机森林在每次节点分裂时随机选择特征，而 Bagging（即使支持随机特征）在整棵树级别选择特征子集。

如果你希望进一步探讨经典 Bagging 和随机森林的数学差异、性能对比，或者需要从头实现决策树以完全摆脱 scikit-learn 依赖，请告诉我！