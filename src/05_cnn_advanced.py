"""
使用卷积神经网络(CNN)识别手写数字
CNN是图像识别的最佳算法！
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查是否安装了深度学习库
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
    print(f"✅ TensorFlow版本: {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("❌ 未安装TensorFlow，正在使用替代方案...")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
    print(f"✅ PyTorch版本: {torch.__version__}")
except ImportError:
    HAS_TORCH = False
    print("❌ 未安装PyTorch")

print("="*80)
print("🚀 卷积神经网络 (CNN) - 图像识别的王者")
print("="*80)

# 加载数据
digits = datasets.load_digits()
X = digits.images  # 保持8x8形状
y = digits.target

print(f"\n📊 数据集信息:")
print(f"   总样本数: {len(X)}")
print(f"   图片尺寸: 8x8")
print(f"   类别数: 10")

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"   训练集: {len(X_train)}")
print(f"   测试集: {len(X_test)}")

# ============ CNN方案1: TensorFlow/Keras ============
if HAS_TF:
    print("\n" + "="*80)
    print("【方案1】使用 TensorFlow/Keras 实现CNN")
    print("="*80)
    
    # 数据预处理
    X_train_tf = X_train.reshape(-1, 8, 8, 1).astype('float32') / 16.0  # 归一化
    X_test_tf = X_test.reshape(-1, 8, 8, 1).astype('float32') / 16.0
    y_train_tf = keras.utils.to_categorical(y_train, 10)  # one-hot编码
    y_test_tf = keras.utils.to_categorical(y_test, 10)
    
    # 构建CNN模型
    print("\n🏗️  构建CNN模型...")
    model_keras = keras.Sequential([
        # 第一层卷积：提取基础特征（边缘、角点）
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=(8, 8, 1), padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # 第二层卷积：提取更复杂的特征（形状、纹理）
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # 展平
        layers.Flatten(),
        
        # 全连接层
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # 防止过拟合
        
        # 输出层
        layers.Dense(10, activation='softmax')
    ])
    
    # 编译模型
    model_keras.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 显示模型结构
    print("\n📐 模型结构:")
    model_keras.summary()
    
    # 训练模型
    print("\n🧠 开始训练CNN...")
    history = model_keras.fit(
        X_train_tf, y_train_tf,
        batch_size=32,
        epochs=50,
        verbose=1,
        validation_data=(X_test_tf, y_test_tf)
    )
    
    # 评估模型
    test_loss, test_acc = model_keras.evaluate(X_test_tf, y_test_tf, verbose=0)
    print(f"\n✅ CNN (TensorFlow) 测试集准确率: {test_acc*100:.2f}%")
    
    # 可视化训练过程
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率曲线
    axes[0].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('准确率', fontsize=12)
    axes[0].set_title('CNN训练过程 - 准确率', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 损失曲线
    axes[1].plot(history.history['loss'], label='训练损失', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='验证损失', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('损失', fontsize=12)
    axes[1].set_title('CNN训练过程 - 损失', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png', dpi=150)
    print("✅ 训练曲线已保存为: cnn_training_history.png")
    plt.show()
    
    # 预测示例
    predictions = model_keras.predict(X_test_tf[:20], verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    # 可视化预测结果
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle('CNN预测结果示例', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i], cmap='gray')
        pred_label = pred_labels[i]
        true_label = y_test[i]
        confidence = predictions[i][pred_label] * 100
        
        title = f'真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.1f}%'
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cnn_predictions.png', dpi=150)
    print("✅ 预测结果已保存为: cnn_predictions.png")
    plt.show()
    
    # 可视化卷积层学到的特征
    print("\n🔍 可视化CNN学到的特征...")
    
    # 提取第一层卷积层的权重
    first_conv_layer = model_keras.layers[0]
    filters, biases = first_conv_layer.get_weights()
    
    # 显示前16个卷积核
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle('CNN第一层卷积核（特征提取器）', fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < min(32, filters.shape[3]):
            ax.imshow(filters[:, :, 0, i], cmap='viridis')
            ax.set_title(f'Filter {i+1}', fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cnn_filters.png', dpi=150)
    print("✅ 卷积核可视化已保存为: cnn_filters.png")
    plt.show()
    
    # 保存模型
    model_keras.save('digit_cnn_model.h5')
    print("\n💾 CNN模型已保存为: digit_cnn_model.h5")

# ============ CNN方案2: PyTorch ============
elif HAS_TORCH:
    print("\n" + "="*80)
    print("【方案2】使用 PyTorch 实现CNN")
    print("="*80)
    
    # 数据预处理
    X_train_pt = torch.FloatTensor(X_train).unsqueeze(1) / 16.0  # 添加通道维度
    X_test_pt = torch.FloatTensor(X_test).unsqueeze(1) / 16.0
    y_train_pt = torch.LongTensor(y_train)
    y_test_pt = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_pt, y_train_pt)
    test_dataset = TensorDataset(X_test_pt, y_test_pt)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 定义CNN模型
    class DigitCNN(nn.Module):
        def __init__(self):
            super(DigitCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 2 * 2, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 2 * 2)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    model_torch = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_torch.parameters(), lr=0.001)
    
    print("\n🧠 开始训练CNN (PyTorch)...")
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(50):
        model_torch.train()
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_torch(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 评估
        model_torch.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model_torch(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        train_losses.append(epoch_loss / len(train_loader))
        test_accuracies.append(acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/50], Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%')
    
    print(f"\n✅ CNN (PyTorch) 最终准确率: {test_accuracies[-1]:.2f}%")
    
    # 可视化训练过程
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(test_accuracies, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('CNN (PyTorch) 训练过程', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cnn_pytorch_training.png', dpi=150)
    plt.show()
    
    # 保存模型
    torch.save(model_torch.state_dict(), 'digit_cnn_pytorch.pth')
    print("\n💾 CNN模型已保存为: digit_cnn_pytorch.pth")

# ============ 如果没有深度学习库 ============
else:
    print("\n" + "="*80)
    print("⚠️  未检测到深度学习库 (TensorFlow 或 PyTorch)")
    print("="*80)
    print("""
要使用CNN，请安装以下其中一个库：

方法1: 安装TensorFlow
    pip install tensorflow

方法2: 安装PyTorch (推荐用于研究)
    # CPU版本
    pip install torch torchvision
    
    # GPU版本 (CUDA支持)
    请访问: https://pytorch.org/get-started/locally/

安装后重新运行此脚本即可！
""")
    
    # 使用scikit-learn的MLP作为替代演示
    print("作为演示，我们使用多层感知机(MLP)模拟...")
    from sklearn.neural_network import MLPClassifier
    from sklearn import metrics
    import time
    
    X_train_flat = X_train.reshape(len(X_train), -1) / 16.0
    X_test_flat = X_test.reshape(len(X_test), -1) / 16.0
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=100,
        random_state=42,
        verbose=True
    )
    
    print("\n训练MLP神经网络...")
    mlp.fit(X_train_flat, y_train)
    
    pred = mlp.predict(X_test_flat)
    acc = metrics.accuracy_score(y_test, pred)
    
    print(f"\n✅ MLP准确率: {acc*100:.2f}%")
    print("\n💡 提示: 安装TensorFlow可以获得更好的CNN实现和更高的准确率！")

# ============ CNN原理解释 ============
print("\n" + "="*80)
print("📚 CNN原理简介")
print("="*80)
print("""
CNN (卷积神经网络) 为什么最适合图像识别？

1️⃣  卷积层 (Convolutional Layer)
   - 自动学习特征提取器（不需要手工设计）
   - 第一层：学习边缘、角点
   - 第二层：学习形状、纹理
   - 第三层：学习更复杂的模式
   
   示例：识别数字"7"
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ 原始图片 │ --> │ 边缘检测 │ --> │ 形状识别 │ --> "这是7"
   │  (8x8)  │     │(3x3卷积)│     │(3x3卷积)│
   └─────────┘     └─────────┘     └─────────┘

2️⃣  池化层 (Pooling Layer)
   - 降低维度，减少计算量
   - 提取最重要的特征
   - 增强模型的鲁棒性（对平移、旋转不敏感）

3️⃣  全连接层 (Fully Connected Layer)
   - 将提取的特征组合起来
   - 最终分类决策

对比传统方法：

KNN:    直接比较像素 → 简单但效果差
SVM:    手工设计特征 → 需要专业知识
CNN:    自动学习特征 → 效果最好！⭐

CNN的优势：
✅ 自动特征学习（不需要人工设计）
✅ 局部连接（关注局部特征）
✅ 参数共享（同一个特征检测器用于整张图）
✅ 平移不变性（数字在哪里都能识别）

在大数据集（如MNIST 28x28, ImageNet）上：
CNN >> 传统机器学习方法
""")

print("\n" + "="*80)
print("🎯 总结")
print("="*80)
print("""
手写数字识别 - 各算法适用场景：

📚 KNN:
   - 用于: 教学、理解机器学习基础
   - 优点: 简单易懂
   - 缺点: 效果差、预测慢
   - 准确率: ~90-95%

🎯 SVM/随机森林:
   - 用于: 中小型数据集、特征明确的任务
   - 优点: 训练快、效果好
   - 缺点: 需要特征工程
   - 准确率: ~96-98%

🧠 MLP神经网络:
   - 用于: 通用机器学习任务
   - 优点: 自动学习特征
   - 缺点: 不如CNN适合图像
   - 准确率: ~97-98%

🚀 CNN:
   - 用于: 图像识别（最佳选择！）
   - 优点: 自动特征学习、准确率最高
   - 缺点: 需要较多数据、训练较慢
   - 准确率: ~99%+

推荐路线：
1. 学习: 从KNN开始理解基础概念
2. 实践: 用SVM/MLP处理实际问题
3. 进阶: 用CNN处理图像任务
4. 高级: 学习ResNet、Transformer等先进架构
""")
print("="*80)

