import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载数据
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

print("="*60)
print("🔬 实验：测试不同的K值对模型精度的影响")
print("="*60)

# 测试不同的K值
k_values = [1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 40, 50]
accuracies = []

for k in k_values:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    accuracies.append(accuracy)
    print(f"K = {k:2d}  →  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 找到最佳K值
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print("\n" + "="*60)
print(f"🏆 最佳K值: K = {best_k}，准确率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("="*60)

# 可视化K值与准确率的关系
plt.figure(figsize=(12, 6))
plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'最佳K={best_k}')
plt.xlabel('K值（邻居数量）', fontsize=14)
plt.ylabel('准确率', fontsize=14)
plt.title('不同K值对KNN模型准确率的影响', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xticks(k_values)

# 标注最高点
plt.scatter([best_k], [best_accuracy], color='red', s=200, zorder=5)
plt.annotate(f'{best_accuracy:.4f}', 
             xy=(best_k, best_accuracy), 
             xytext=(best_k+5, best_accuracy-0.01),
             fontsize=12, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.show()

print("\n📚 观察结论：")
print("  - K值不是越大越好，也不是越小越好")
print("  - K太小容易受噪声影响（过拟合）")
print("  - K太大会包含不相关的邻居（欠拟合）")
print("  - 需要通过实验找到最佳K值")

