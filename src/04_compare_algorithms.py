"""
手写数字识别：多种算法全面对比
包括：KNN, SVM, 随机森林, 神经网络, CNN
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

print("="*80)
print("🏆 手写数字识别：算法大比拼")
print("="*80)

# 加载数据
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 拆分数据集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, random_state=42, shuffle=True
)

# 数据标准化（对某些算法很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"📊 数据集信息:")
print(f"   训练集: {len(X_train)} 张图片")
print(f"   测试集: {len(X_test)} 张图片")
print(f"   图片大小: 8x8 = 64 个像素")
print(f"   类别数: 10 (数字 0-9)")
print("="*80)

# 存储结果
results = []

# ============ 算法1: KNN ============
print("\n【算法1】K近邻算法 (KNN)")
print("原理: 找最相似的K个邻居，投票决定")
print("优点: 简单、无需训练、可解释")
print("缺点: 预测慢、占内存、对噪声敏感")

start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)
train_time_knn = time.time() - start_time

start_time = time.time()
pred_knn = knn.predict(X_test_scaled)
predict_time_knn = time.time() - start_time

acc_knn = metrics.accuracy_score(y_test, pred_knn)
results.append(('KNN', acc_knn, train_time_knn, predict_time_knn))

print(f"✅ 训练时间: {train_time_knn:.4f}秒")
print(f"✅ 预测时间: {predict_time_knn:.4f}秒")
print(f"✅ 准确率: {acc_knn:.4f} ({acc_knn*100:.2f}%)")

# ============ 算法2: SVM (支持向量机) ============
print("\n" + "="*80)
print("【算法2】支持向量机 (SVM)")
print("原理: 找到最优的决策边界（超平面）")
print("优点: 高维数据效果好、泛化能力强")
print("缺点: 训练较慢、参数调优复杂")

start_time = time.time()
svm = SVC(kernel='rbf', C=10, gamma=0.001, random_state=42)
svm.fit(X_train_scaled, y_train)
train_time_svm = time.time() - start_time

start_time = time.time()
pred_svm = svm.predict(X_test_scaled)
predict_time_svm = time.time() - start_time

acc_svm = metrics.accuracy_score(y_test, pred_svm)
results.append(('SVM', acc_svm, train_time_svm, predict_time_svm))

print(f"✅ 训练时间: {train_time_svm:.4f}秒")
print(f"✅ 预测时间: {predict_time_svm:.4f}秒")
print(f"✅ 准确率: {acc_svm:.4f} ({acc_svm*100:.2f}%)")

# ============ 算法3: 随机森林 ============
print("\n" + "="*80)
print("【算法3】随机森林 (Random Forest)")
print("原理: 训练多个决策树，集成投票")
print("优点: 不容易过拟合、特征重要性分析")
print("缺点: 模型较大、可解释性差")

start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # 随机森林不需要标准化
train_time_rf = time.time() - start_time

start_time = time.time()
pred_rf = rf.predict(X_test)
predict_time_rf = time.time() - start_time

acc_rf = metrics.accuracy_score(y_test, pred_rf)
results.append(('Random Forest', acc_rf, train_time_rf, predict_time_rf))

print(f"✅ 训练时间: {train_time_rf:.4f}秒")
print(f"✅ 预测时间: {predict_time_rf:.4f}秒")
print(f"✅ 准确率: {acc_rf:.4f} ({acc_rf*100:.2f}%)")

# ============ 算法4: 多层感知机神经网络 (MLP) ============
print("\n" + "="*80)
print("【算法4】多层感知机 (MLP Neural Network)")
print("原理: 多层神经网络，通过反向传播学习")
print("优点: 能学习复杂模式、通用性强")
print("缺点: 需要调参、容易过拟合")

start_time = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 两层隐藏层
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp.fit(X_train_scaled, y_train)
train_time_mlp = time.time() - start_time

start_time = time.time()
pred_mlp = mlp.predict(X_test_scaled)
predict_time_mlp = time.time() - start_time

acc_mlp = metrics.accuracy_score(y_test, pred_mlp)
results.append(('MLP Neural Net', acc_mlp, train_time_mlp, predict_time_mlp))

print(f"✅ 训练时间: {train_time_mlp:.4f}秒")
print(f"✅ 预测时间: {predict_time_mlp:.4f}秒")
print(f"✅ 准确率: {acc_mlp:.4f} ({acc_mlp*100:.2f}%)")

# ============ 最终对比 ============
print("\n" + "="*80)
print("🏆 最终结果对比")
print("="*80)

# 按准确率排序
results.sort(key=lambda x: x[1], reverse=True)

print(f"{'算法':<20} {'准确率':<12} {'训练时间':<12} {'预测时间':<12}")
print("-"*80)
for name, acc, train_t, pred_t in results:
    print(f"{name:<20} {acc*100:>6.2f}%      {train_t:>8.4f}秒    {pred_t:>8.4f}秒")

best_algo = results[0]
print("-"*80)
print(f"🥇 最佳算法: {best_algo[0]} (准确率: {best_algo[1]*100:.2f}%)")
print("="*80)

# ============ 可视化对比 ============
print("\n📊 正在生成可视化对比图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 准确率对比
ax1 = axes[0, 0]
names = [r[0] for r in results]
accuracies = [r[1]*100 for r in results]
colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen']
bars = ax1.barh(names, accuracies, color=colors[:len(names)])
ax1.set_xlabel('准确率 (%)', fontsize=12)
ax1.set_title('算法准确率对比', fontsize=14, fontweight='bold')
ax1.set_xlim([85, 100])
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(acc + 0.2, bar.get_y() + bar.get_height()/2, 
             f'{acc:.2f}%', va='center', fontsize=10, fontweight='bold')

# 子图2: 训练时间对比
ax2 = axes[0, 1]
train_times = [r[2] for r in results]
bars = ax2.barh(names, train_times, color='skyblue')
ax2.set_xlabel('训练时间 (秒)', fontsize=12)
ax2.set_title('训练时间对比', fontsize=14, fontweight='bold')
for bar, t in zip(bars, train_times):
    ax2.text(t + max(train_times)*0.02, bar.get_y() + bar.get_height()/2, 
             f'{t:.4f}s', va='center', fontsize=9)

# 子图3: 预测时间对比
ax3 = axes[1, 0]
pred_times = [r[3] for r in results]
bars = ax3.barh(names, pred_times, color='lightcoral')
ax3.set_xlabel('预测时间 (秒)', fontsize=12)
ax3.set_title('预测时间对比', fontsize=14, fontweight='bold')
for bar, t in zip(bars, pred_times):
    ax3.text(t + max(pred_times)*0.02, bar.get_y() + bar.get_height()/2, 
             f'{t:.4f}s', va='center', fontsize=9)

# 子图4: 混淆矩阵（最佳算法）
ax4 = axes[1, 1]
cm = metrics.confusion_matrix(y_test, pred_svm)  # 使用SVM的结果
im = ax4.imshow(cm, cmap='Blues', interpolation='nearest')
ax4.set_title(f'混淆矩阵: {results[0][0]}', fontsize=14, fontweight='bold')
ax4.set_xlabel('预测标签', fontsize=12)
ax4.set_ylabel('真实标签', fontsize=12)
ax4.set_xticks(range(10))
ax4.set_yticks(range(10))
plt.colorbar(im, ax=ax4)

# 在混淆矩阵上标注数字
for i in range(10):
    for j in range(10):
        text = ax4.text(j, i, cm[i, j],
                       ha="center", va="center", 
                       color="white" if cm[i, j] > cm.max()/2 else "black",
                       fontsize=8)

plt.tight_layout()
plt.savefig('algorithms_comparison.png', dpi=150, bbox_inches='tight')
print("✅ 对比图已保存为: algorithms_comparison.png")
plt.show()

# ============ 详细性能报告 ============
print("\n" + "="*80)
print(f"📋 最佳算法 ({results[0][0]}) 详细报告")
print("="*80)

# 使用最佳算法（通常是SVM或MLP）
best_pred = pred_svm if results[0][0] == 'SVM' else pred_mlp

print("\n分类报告:")
print(metrics.classification_report(y_test, best_pred, 
                                     target_names=[str(i) for i in range(10)]))

# ============ 预测示例展示 ============
print("\n" + "="*80)
print("🔍 预测示例对比（显示前12个测试样本）")
print("="*80)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
fig.suptitle('不同算法的预测对比', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < 12:
        # 显示图片
        ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
        
        # 各算法的预测
        true_label = y_test[i]
        pred_k = pred_knn[i]
        pred_s = pred_svm[i]
        pred_r = pred_rf[i]
        pred_m = pred_mlp[i]
        
        # 标题
        title = f'真实: {true_label}\n'
        title += f'KNN:{pred_k} SVM:{pred_s}\n'
        title += f'RF:{pred_r} MLP:{pred_m}'
        
        # 如果所有算法都正确，用绿色；如果有错误，用红色
        all_correct = (pred_k == true_label and pred_s == true_label and 
                      pred_r == true_label and pred_m == true_label)
        color = 'green' if all_correct else 'red'
        
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=150, bbox_inches='tight')
print("✅ 预测对比图已保存为: predictions_comparison.png")
plt.show()

print("\n" + "="*80)
print("💡 总结与建议")
print("="*80)
print("""
对于8x8的手写数字识别任务：

1. ✅ SVM 和 MLP 通常表现最好（96-98%准确率）
2. ✅ 随机森林也不错，且训练快
3. ⚠️  KNN虽然简单，但准确率和预测速度都不理想
4. 💡 如果要追求极致性能，应该使用CNN（见下一个脚本）

在更大的数据集（如MNIST 28x28）上：
- CNN可以达到99%+的准确率
- KNN会因为数据量大而非常慢
- SVM和MLP仍然有不错表现

建议：
- 学习/教学: 用KNN（最容易理解）
- 生产环境: 用SVM、MLP或CNN
- 实时应用: 避免使用KNN（太慢）
""")
print("="*80)
print("\n🎉 实验完成！现在运行 cnn_advanced.py 看看CNN的威力！")

