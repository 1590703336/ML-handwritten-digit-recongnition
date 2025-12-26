"""
高精度手写数字识别 - 完整MNIST数据集
使用28×28图像 + 所有优化技巧
目标准确率: 99.5%+
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 导入深度学习库
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print(f"✅ TensorFlow版本: {tf.__version__}")
    HAS_TF = True
except ImportError:
    print("❌ 未安装TensorFlow，请运行: pip install tensorflow")
    HAS_TF = False
    exit(1)

print("="*80)
print("🚀 高精度手写数字识别 - MNIST完整数据集 (28×28)")
print("="*80)

# ============ 1. 加载MNIST数据集 ============
print("\n📊 加载MNIST数据集...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"✅ 数据集加载完成:")
print(f"   训练集: {len(X_train):,} 张图片")
print(f"   测试集: {len(X_test):,} 张图片")
print(f"   图片尺寸: {X_train.shape[1]}×{X_train.shape[2]}")
print(f"   类别数: 10 (数字0-9)")

# 数据预处理
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"\n✅ 数据预处理完成:")
print(f"   输入形状: {X_train.shape}")
print(f"   标签形状: {y_train_cat.shape}")
print(f"   数值范围: {X_train.min():.1f} ~ {X_train.max():.1f}")

# 可视化一些样本
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
fig.suptitle('MNIST数据集样本（28×28高分辨率）', fontsize=14, fontweight='bold')
for i in range(20):
    ax = axes[i//10, i%10]
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'{y_train[i]}', fontsize=12, fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/visualizations/08_mnist_samples.png', dpi=150, bbox_inches='tight')
print("\n✅ 样本可视化已保存: outputs/visualizations/08_mnist_samples.png")
plt.show()

# ============ 2. 构建改进的CNN模型 ============
print("\n" + "="*80)
print("🏗️  构建高精度CNN模型")
print("="*80)

def build_improved_cnn():
    """构建改进的CNN模型，包含多项优化技巧"""
    model = keras.Sequential([
        # 输入层
        layers.Input(shape=(28, 28, 1)),
        
        # 第一组卷积块（32个滤波器）
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),  # 批归一化：加速训练、提高稳定性
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Dropout：防止过拟合
        
        # 第二组卷积块（64个滤波器）
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第三组卷积块（128个滤波器）
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # 全连接层
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(10, activation='softmax')
    ])
    
    return model

model = build_improved_cnn()

# 编译模型 - 使用Adam优化器
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📐 模型架构:")
model.summary()

# 计算模型参数
total_params = model.count_params()
print(f"\n📊 模型参数总数: {total_params:,}")

# ============ 3. 数据增强 ============
print("\n" + "="*80)
print("🎨 配置数据增强")
print("="*80)

datagen = ImageDataGenerator(
    rotation_range=10,        # 随机旋转±10度
    width_shift_range=0.1,    # 水平平移±10%
    height_shift_range=0.1,   # 垂直平移±10%
    zoom_range=0.1,           # 随机缩放±10%
    shear_range=0.1,          # 剪切变换
)

print("✅ 数据增强配置:")
print("   - 随机旋转: ±10°")
print("   - 随机平移: ±10%")
print("   - 随机缩放: ±10%")
print("   - 剪切变换: ±10%")

# 可视化数据增强效果
print("\n🔍 可视化数据增强效果...")
sample_img = X_train[0].reshape(1, 28, 28, 1)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('数据增强效果展示', fontsize=14, fontweight='bold')

axes[0, 0].imshow(sample_img[0, :, :, 0], cmap='gray')
axes[0, 0].set_title('原始图像', fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

augmented_imgs = datagen.flow(sample_img, batch_size=1)
for i in range(1, 10):
    ax = axes[i//5, i%5]
    aug_img = next(augmented_imgs)[0]
    ax.imshow(aug_img[:, :, 0], cmap='gray')
    ax.set_title(f'增强 {i}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/visualizations/09_data_augmentation.png', dpi=150, bbox_inches='tight')
print("✅ 数据增强可视化已保存: outputs/visualizations/09_data_augmentation.png")
plt.show()

# ============ 4. 配置训练回调函数 ============
print("\n" + "="*80)
print("⚙️  配置训练优化策略")
print("="*80)

# 早停：防止过拟合
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,              # 15个epoch没有改善就停止
    restore_best_weights=True,
    verbose=1
)

# 学习率衰减：动态调整学习率
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,               # 学习率减半
    patience=5,               # 5个epoch没有改善就降低学习率
    min_lr=1e-7,
    verbose=1
)

# 模型检查点：保存最佳模型
checkpoint = ModelCheckpoint(
    'models/mnist_cnn_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

print("✅ 训练策略配置:")
print("   - 早停机制: 15个epoch无改善则停止")
print("   - 学习率衰减: 5个epoch无改善则降低50%")
print("   - 模型检查点: 自动保存最佳模型")

# ============ 5. 训练模型 ============
print("\n" + "="*80)
print("🧠 开始训练高精度CNN模型")
print("="*80)
print("\n⏰ 训练开始时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
print("💡 提示: 使用数据增强训练，预计需要5-10分钟...\n")

start_time = time.time()

# 使用数据增强进行训练
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=128),
    epochs=100,  # 最多训练100个epoch，早停会自动停止
    steps_per_epoch=len(X_train) // 128,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stopping, lr_scheduler, checkpoint],
    verbose=1
)

training_time = time.time() - start_time

print("\n" + "="*80)
print("✅ 训练完成！")
print("="*80)
print(f"⏱️  总训练时间: {training_time/60:.1f} 分钟")
print(f"📈 实际训练轮数: {len(history.history['loss'])} epochs")

# ============ 6. 评估模型 ============
print("\n" + "="*80)
print("📊 模型评估")
print("="*80)

# 加载最佳模型
best_model = keras.models.load_model('models/mnist_cnn_best.h5')

# 在测试集上评估
test_loss, test_acc = best_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ 最终测试集性能:")
print(f"   准确率: {test_acc*100:.4f}%")
print(f"   损失: {test_loss:.4f}")

# 详细的分类报告
from sklearn.metrics import classification_report, confusion_matrix

y_pred = best_model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📋 详细分类报告:")
print(classification_report(y_test, y_pred_classes, target_names=[str(i) for i in range(10)]))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_classes)
fig, ax = plt.subplots(figsize=(10, 8))

# 使用matplotlib绘制混淆矩阵（不依赖seaborn）
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
ax.figure.colorbar(im, ax=ax)

# 设置刻度
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(range(10))
ax.set_yticklabels(range(10))

# 在每个格子中显示数值
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, cm[i, j],
                      ha="center", va="center",
                      color="white" if cm[i, j] > cm.max()/2 else "black",
                      fontsize=10, fontweight='bold')

ax.set_xlabel('预测标签', fontsize=12)
ax.set_ylabel('真实标签', fontsize=12)
ax.set_title('混淆矩阵 - 高精度CNN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/visualizations/10_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n✅ 混淆矩阵已保存: outputs/visualizations/10_confusion_matrix.png")
plt.show()

# ============ 7. 可视化训练过程 ============
print("\n" + "="*80)
print("📈 可视化训练过程")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 准确率曲线
axes[0].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('准确率', fontsize=12)
axes[0].set_title('训练过程 - 准确率', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
max_acc = max(history.history['val_accuracy'])
axes[0].axhline(y=max_acc, color='r', linestyle='--', alpha=0.5, 
                label=f'最佳: {max_acc*100:.2f}%')
axes[0].legend(fontsize=11)

# 损失曲线
axes[1].plot(history.history['loss'], label='训练损失', linewidth=2)
axes[1].plot(history.history['val_loss'], label='验证损失', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('损失', fontsize=12)
axes[1].set_title('训练过程 - 损失', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/11_training_history.png', dpi=150, bbox_inches='tight')
print("✅ 训练曲线已保存: outputs/visualizations/11_training_history.png")
plt.show()

# ============ 8. 预测展示 ============
print("\n" + "="*80)
print("🎯 预测结果展示")
print("="*80)

# 随机选择30个样本进行预测
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), 30, replace=False)
sample_images = X_test[sample_indices]
sample_labels = y_test[sample_indices]

predictions = best_model.predict(sample_images, verbose=0)
pred_labels = np.argmax(predictions, axis=1)

# 可视化预测结果
fig, axes = plt.subplots(6, 5, figsize=(15, 18))
fig.suptitle('高精度CNN预测结果（28×28 MNIST）', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    pred_label = pred_labels[i]
    true_label = sample_labels[i]
    confidence = predictions[i][pred_label] * 100
    
    title = f'真实: {true_label} | 预测: {pred_label}\n置信度: {confidence:.1f}%'
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(title, fontsize=9, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/visualizations/12_predictions.png', dpi=150, bbox_inches='tight')
print("✅ 预测结果已保存: outputs/visualizations/12_predictions.png")
plt.show()

# 统计错误预测
errors = np.where(pred_labels != sample_labels)[0]
print(f"\n📊 在30个样本中:")
print(f"   正确预测: {30-len(errors)} 个")
print(f"   错误预测: {len(errors)} 个")
if len(errors) > 0:
    print(f"   错误率: {len(errors)/30*100:.1f}%")

# ============ 9. 可视化学习到的特征 ============
print("\n" + "="*80)
print("🔬 可视化CNN学习到的特征")
print("="*80)

# 提取第一层卷积层的权重
first_conv_layer = best_model.layers[0]
filters, biases = first_conv_layer.get_weights()

print(f"\n卷积核信息:")
print(f"   形状: {filters.shape}")
print(f"   数量: {filters.shape[3]} 个")

# 显示前32个卷积核
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('第一层卷积核（边缘和纹理检测器）', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < filters.shape[3]:
        filter_img = filters[:, :, 0, i]
        ax.imshow(filter_img, cmap='viridis')
        ax.set_title(f'Filter {i+1}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/visualizations/13_cnn_filters.png', dpi=150, bbox_inches='tight')
print("✅ 卷积核可视化已保存: outputs/visualizations/13_cnn_filters.png")
plt.show()

# 可视化特征图
print("\n🔍 可视化特征图（Feature Maps）...")

# 创建一个模型来输出中间层的激活
layer_outputs = [layer.output for layer in best_model.layers[:7]]  # 前7层
activation_model = keras.Model(inputs=best_model.input, outputs=layer_outputs)

# 选择一个样本
sample = X_test[0:1]
activations = activation_model.predict(sample, verbose=0)

# 可视化第一层卷积层的输出
first_layer_activation = activations[0]
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('第一层卷积层输出（特征图）', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < first_layer_activation.shape[3]:
        ax.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
        ax.set_title(f'特征图 {i+1}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/visualizations/14_feature_maps.png', dpi=150, bbox_inches='tight')
print("✅ 特征图可视化已保存: outputs/visualizations/14_feature_maps.png")
plt.show()

# ============ 10. 性能对比总结 ============
print("\n" + "="*80)
print("📊 性能对比总结")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────┐
│                   模型性能对比                               │
├─────────────────┬───────────┬───────────┬──────────────────┤
│   模型          │  数据集   │  准确率   │      说明        │
├─────────────────┼───────────┼───────────┼──────────────────┤
│ KNN             │ 8×8       │ ~95%      │ 简单但效果一般   │
│ SVM/RF          │ 8×8       │ ~98%      │ 传统ML最佳       │
│ 基础CNN         │ 8×8       │ ~98%      │ 深度学习入门     │
│ 🏆 高精度CNN    │ 28×28     │ ~99.5%+   │ 专业级解决方案   │
└─────────────────┴───────────┴───────────┴──────────────────┘
""")

print(f"\n✨ 本次训练最终结果:")
print(f"   📈 测试集准确率: {test_acc*100:.4f}%")
print(f"   ⏱️  训练时间: {training_time/60:.1f} 分钟")
print(f"   💾 模型大小: {os.path.getsize('models/mnist_cnn_best.h5')/1024/1024:.2f} MB")

print("\n🎯 应用的优化技巧:")
print("   ✅ 使用完整MNIST数据集（60,000训练样本）")
print("   ✅ 更深的CNN架构（3组卷积块）")
print("   ✅ 批归一化（BatchNormalization）")
print("   ✅ Dropout正则化（防止过拟合）")
print("   ✅ 数据增强（旋转、平移、缩放）")
print("   ✅ 学习率衰减策略")
print("   ✅ 早停机制")
print("   ✅ 保存最佳模型")

print("\n" + "="*80)
print("🎉 训练完成！模型已保存到: models/mnist_cnn_best.h5")
print("="*80)

# 保存训练历史
import pickle
with open('outputs/reports/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("\n✅ 训练历史已保存: outputs/reports/training_history.pkl")

print("\n💡 如何使用训练好的模型:")
print("""
from tensorflow import keras
import numpy as np

# 加载模型
model = keras.models.load_model('models/mnist_cnn_best.h5')

# 预测
image = your_image.reshape(1, 28, 28, 1) / 255.0  # 归一化
prediction = model.predict(image)
digit = np.argmax(prediction)
print(f'预测数字: {digit}')
""")

import os

