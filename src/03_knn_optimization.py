import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time

print("="*70)
print("🚀 提升KNN模型精度的多种方法")
print("="*70)

# 加载数据
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# ============ 方法1: 基线模型（原始数据，K=3） ============
print("\n【方法1】基线模型：原始数据 + K=3")
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

classifier1 = KNeighborsClassifier(n_neighbors=3)
classifier1.fit(X_train, y_train)
pred1 = classifier1.predict(X_test)
acc1 = metrics.accuracy_score(y_test, pred1)
print(f"  准确率: {acc1:.4f} ({acc1*100:.2f}%)")

# ============ 方法2: 优化K值 ============
print("\n【方法2】优化K值：寻找最佳K")
best_k = 1
best_acc = 0
for k in range(1, 31):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, pred)
    if acc > best_acc:
        best_acc = acc
        best_k = k

classifier2 = KNeighborsClassifier(n_neighbors=best_k)
classifier2.fit(X_train, y_train)
pred2 = classifier2.predict(X_test)
acc2 = metrics.accuracy_score(y_test, pred2)
print(f"  最佳K值: {best_k}")
print(f"  准确率: {acc2:.4f} ({acc2*100:.2f}%) ↑ 提升: {(acc2-acc1)*100:.2f}%")

# ============ 方法3: 数据标准化（归一化） ============
print("\n【方法3】数据标准化：让每个特征的权重相同")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier3 = KNeighborsClassifier(n_neighbors=3)
classifier3.fit(X_train_scaled, y_train)
pred3 = classifier3.predict(X_test_scaled)
acc3 = metrics.accuracy_score(y_test, pred3)
print(f"  准确率: {acc3:.4f} ({acc3*100:.2f}%) ↑ 提升: {(acc3-acc1)*100:.2f}%")

# ============ 方法4: 改变距离度量方式 ============
print("\n【方法4】改变距离度量：测试不同的距离计算方法")
# 欧氏距离 (默认)
clf_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
clf_euclidean.fit(X_train, y_train)
pred_euc = clf_euclidean.predict(X_test)
acc_euc = metrics.accuracy_score(y_test, pred_euc)
print(f"  欧氏距离: {acc_euc:.4f} ({acc_euc*100:.2f}%)")

# 曼哈顿距离
clf_manhattan = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
clf_manhattan.fit(X_train, y_train)
pred_man = clf_manhattan.predict(X_test)
acc_man = metrics.accuracy_score(y_test, pred_man)
print(f"  曼哈顿距离: {acc_man:.4f} ({acc_man*100:.2f}%)")

# 闵可夫斯基距离
clf_minkowski = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3)
clf_minkowski.fit(X_train, y_train)
pred_min = clf_minkowski.predict(X_test)
acc_min = metrics.accuracy_score(y_test, pred_min)
print(f"  闵可夫斯基距离: {acc_min:.4f} ({acc_min*100:.2f}%)")

best_metric_acc = max(acc_euc, acc_man, acc_min)
print(f"  最佳距离度量提升: {(best_metric_acc-acc1)*100:.2f}%")

# ============ 方法5: 增加训练数据（更大的训练集） ============
print("\n【方法5】增加训练数据：从50%增加到80%")
X_train_large, X_test_small, y_train_large, y_test_small = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

classifier5 = KNeighborsClassifier(n_neighbors=3)
classifier5.fit(X_train_large, y_train_large)
pred5 = classifier5.predict(X_test_small)
acc5 = metrics.accuracy_score(y_test_small, pred5)
print(f"  训练集大小: {len(X_train_large)} 张（原来是 {len(X_train)} 张）")
print(f"  准确率: {acc5:.4f} ({acc5*100:.2f}%)")
print(f"  注意：测试集变小了，所以不能直接比较")

# ============ 方法6: 加权投票（距离越近权重越大） ============
print("\n【方法6】加权投票：距离近的邻居权重更大")
classifier6 = KNeighborsClassifier(n_neighbors=5, weights='distance')
classifier6.fit(X_train, y_train)
pred6 = classifier6.predict(X_test)
acc6 = metrics.accuracy_score(y_test, pred6)
print(f"  准确率: {acc6:.4f} ({acc6*100:.2f}%) ↑ 提升: {(acc6-acc1)*100:.2f}%")

# ============ 方法7: 组合多种优化（终极版本） ============
print("\n【方法7】组合优化：标准化 + 最佳K + 加权投票 + 最佳距离")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier7 = KNeighborsClassifier(
    n_neighbors=best_k, 
    weights='distance',
    metric='manhattan'
)
classifier7.fit(X_train_scaled, y_train)
pred7 = classifier7.predict(X_test_scaled)
acc7 = metrics.accuracy_score(y_test, pred7)
print(f"  准确率: {acc7:.4f} ({acc7*100:.2f}%) ↑ 提升: {(acc7-acc1)*100:.2f}%")

# ============ 总结对比 ============
print("\n" + "="*70)
print("📊 各方法准确率对比总结：")
print("="*70)

methods = [
    '方法1: 基线模型（K=3）',
    f'方法2: 优化K值（K={best_k}）',
    '方法3: 数据标准化',
    '方法6: 加权投票',
    '方法7: 组合优化'
]
accuracies = [acc1, acc2, acc3, acc6, acc7]
improvements = [0, (acc2-acc1)*100, (acc3-acc1)*100, (acc6-acc1)*100, (acc7-acc1)*100]

for method, acc, imp in zip(methods, accuracies, improvements):
    if imp > 0:
        print(f"{method:30s} → {acc:.4f} ({acc*100:.2f}%) [+{imp:.2f}%]")
    else:
        print(f"{method:30s} → {acc:.4f} ({acc*100:.2f}%)")

print("\n" + "="*70)
print("💡 提升精度的方法总结：")
print("="*70)
print("1. ✅ 优化K值 - 通过实验找到最佳K")
print("2. ✅ 数据标准化 - 让不同特征的权重平衡")
print("3. ✅ 改变距离度量 - 尝试不同的距离计算方法")
print("4. ✅ 增加训练数据 - 更多数据通常能提升精度")
print("5. ✅ 加权投票 - 距离近的邻居权重更大")
print("6. ✅ 数据增强 - 对图片进行旋转、平移等操作")
print("7. ✅ 特征工程 - 提取更有意义的特征")
print("8. ✅ 使用更强大的算法 - 如SVM、随机森林、神经网络")
print("="*70)

# 可视化对比
plt.figure(figsize=(14, 6))
colors = ['gray', 'blue', 'green', 'orange', 'red']
bars = plt.bar(range(len(methods)), [a*100 for a in accuracies], color=colors, alpha=0.7)

# 在柱子上标注准确率
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc*100:.2f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('优化方法', fontsize=13)
plt.ylabel('准确率 (%)', fontsize=13)
plt.title('不同优化方法对KNN模型准确率的提升效果', fontsize=15, fontweight='bold')
plt.xticks(range(len(methods)), 
           ['基线\n(K=3)', f'优化K\n(K={best_k})', '标准化', '加权\n投票', '组合\n优化'],
           fontsize=11)
plt.ylim([min(accuracies)*100-2, 100])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n🎉 实验完成！现在你知道如何提升KNN模型的精度了！")

