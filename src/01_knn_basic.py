import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib # 用来保存训练好的模型
import time

# 1. 获取数据 (Data Ingestion)
# 加载 8x8 的数字数据集
digits = datasets.load_digits()

# 2. 数据预处理 (Preprocessing)
# 图片本质是 8x8 矩阵，但模型需要一维向量 (Vector)。
# 这里的 -1 意味着让 numpy 自动计算维度，把 (n, 8, 8) 变成 (n, 64)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 3. 拆分数据集 (Train/Test Split)
# 50% 学习，50% 考试。
# random_state=42 是为了保证每次运行结果一致 (Reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# ============ 可视化训练数据 ============
print(f"📊 训练集共有 {len(X_train)} 张图片")
print(f"📊 测试集共有 {len(X_test)} 张图片\n")

# 选项1: 显示前100张训练图片（网格形式）
print("正在显示前100张训练图片...")
fig, axes = plt.subplots(10, 10, figsize=(12, 12))
fig.suptitle('前100张训练图片（KNN会记住这些图）', fontsize=16)

for i, (ax, image, label) in enumerate(zip(axes.flat, X_train[:100], y_train[:100])):
    # 将一维数据重新变成8x8图片
    ax.imshow(image.reshape(8, 8), cmap='gray')
    ax.set_title(f'{label}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()

# 选项2: 逐张显示训练图片（可设置显示数量）
display_count = int(input("\n你想逐张查看多少张训练图片？(建议20-50，输入0跳过): "))

if display_count > 0:
    display_count = min(display_count, len(X_train))  # 不超过训练集大小
    print(f"\n开始逐张显示 {display_count} 张训练图片...")
    
    # 创建交互式窗口
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(4, 4))
    
    for i in range(display_count):
        ax.clear()
        ax.imshow(X_train[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'训练图片 {i+1}/{display_count} - 标签: {y_train[i]}', fontsize=14)
        ax.axis('off')
        plt.pause(0.1)  # 暂停0.1秒，可以调整速度
    
    plt.ioff()  # 关闭交互模式
    plt.close()
    
print("\n" + "="*50)

# 4. 定义模型 (Model Definition)
# 这里我们用 K=3 的 KNN 算法
classifier = KNeighborsClassifier(n_neighbors=3)

# 5. 训练 (Training/Fitting)
print("正在训练 KNN 模型... 🧠")
print("⚠️  注意：KNN的'训练'只是把数据存起来，不做复杂计算！")
start_time = time.time()
classifier.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"✅ 训练完成！用时: {training_time:.4f}秒（很快吧？因为只是存储数据）")


# 6. 评估 (Evaluation)
print("\n正在测试模型... 📝")
print("⚠️  预测时才是KNN真正工作的时候！每次预测都要计算距离...")
start_time = time.time()
predicted = classifier.predict(X_test)
prediction_time = time.time() - start_time
print(f"✅ 预测完成！用时: {prediction_time:.4f}秒")
print(f"📊 模型准确率: {metrics.accuracy_score(y_test, predicted):.2%}")

# 显示一些预测示例
print("\n" + "="*50)
print("🔍 让我们看看KNN是如何预测的（显示前9个测试样本）...")
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle('KNN预测示例：找最近的3个邻居投票', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < 9:
        # 显示测试图片
        ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
        
        # 找到最近的3个邻居
        distances, indices = classifier.kneighbors(X_test[i].reshape(1, -1), n_neighbors=3)
        neighbors_labels = y_train[indices[0]]
        
        # 标题显示真实标签、预测标签和3个邻居
        title = f'真实:{y_test[i]} 预测:{predicted[i]}\n'
        title += f'3个邻居: {neighbors_labels[0]}, {neighbors_labels[1]}, {neighbors_labels[2]}'
        color = 'green' if y_test[i] == predicted[i] else 'red'
        ax.set_title(title, fontsize=10, color=color)
        ax.axis('off')

plt.tight_layout()
plt.show()

# 7. 保存模型 (Save Model)
# 这样下次你就不用重新训练了，直接读取 'digit_model.pkl' 即可
joblib.dump(classifier, 'digit_model.pkl')
print("\n💾 模型已保存为 'digit_model.pkl' ✅")
print("\n" + "="*50)
print("📚 KNN算法总结：")
print("  - 训练阶段：只是存储数据（超快）")
print("  - 预测阶段：计算距离+找邻居+投票（较慢）")
print("  - 优点：简单、无需训练")
print("  - 缺点：预测慢、占内存")
print("="*50)