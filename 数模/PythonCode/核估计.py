import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge

# 设置随机种子，确保实验可重复性
np.random.seed(0)

# 生成模拟数据集
n_samples = 100 # 数据样本数量
n_features = 10 # 每个样本的特征数量
X = np.random.rand(n_samples, n_features) # 根据均匀分布生成n_samples个样本，每个样本有n_features个特征
true_w = np.random.randn(n_features) # 随机生成真实的权重向量
noise = np.random.randn(n_samples) # 生成随机噪声项，模拟真实响应上的噪声
y = np.dot(X, true_w) + noise # 计算响应变量，通过X乘以真实权重加上噪声得到

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 使用sklearn库的train_test_split函数，将整个数据集划分为训练集和测试集
# 这里指定测试集占总样本数的20%，random_state参数确保划分过程可重复

# 创建并训练核回归模型
kr = KernelRidge(alpha=1.0, kernel='rbf')
# 初始化一个KernelRidge对象，这是一种基于核方法的回归模型
# 参数alpha表示正则化强度，kernel='rbf'表示使用径向基函数作为内核

kr.fit(X_train, y_train)
# 使用训练集(X_train, y_train)来训练核回归模型

