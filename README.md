# 手写数字识别 - 从KNN到CNN的完整教程

[English](#english-version) | [中文](#chinese-version)

## Chinese Version

一个完整的机器学习入门项目，通过手写数字识别，系统学习从传统机器学习（KNN）到深度学习（CNN）的各种算法。

### 📊 项目概览

本项目实现了5种不同的机器学习算法来识别手写数字：
- **KNN** (K-Nearest Neighbors) - 最简单的入门算法
- **SVM** (Support Vector Machine) - 高效的分类器
- **Random Forest** - 集成学习方法
- **MLP** (Multi-Layer Perceptron) - 传统神经网络
- **CNN** (Convolutional Neural Network) - 最适合图像的深度学习模型

### 🎯 算法性能对比

| 算法 | 数据集 | 准确率 | 训练时间 | 推荐场景 |
|------|--------|--------|---------|---------|
| KNN | 8×8 | ~92% | 0.001秒 | 教学、理解基础 |
| SVM | 8×8 | ~97% | 0.35秒 | 生产环境 |
| Random Forest | 8×8 | ~95% | 0.12秒 | 通用任务 |
| MLP | 8×8 | ~98% | 12秒 | 通用深度学习 |
| CNN (基础) | 8×8 | ~98% | 45秒 | 图像识别入门 |
| **CNN (高精度)** | **28×28 MNIST** | **99.68%** | **17分钟** | **专业级应用** ⭐ |

### 📁 项目结构

```
handwritten-digit-recognition/
├── README.md                          # 项目主文档
├── requirements.txt                   # Python依赖
├── .gitignore                        # Git忽略文件
├── test_cnn_images.py                # 🆕 测试脚本（使用训练好的模型）
│
├── src/                              # 源代码目录
│   ├── 01_knn_basic.py              # KNN基础训练（入门必看）
│   ├── 02_knn_tuning.py             # KNN调参实验
│   ├── 03_knn_optimization.py       # KNN优化方法
│   ├── 04_compare_algorithms.py     # 多算法对比
│   ├── 05_cnn_advanced.py           # CNN实现（8×8数据集）
│   ├── 06_cnn_mnist_advanced.py     # 🆕 高精度CNN（28×28 MNIST）
│   ├── 06_verify_filters.py         # 验证卷积核差异
│   └── predict.py                    # 使用训练好的模型预测
│
├── docs/                             # 文档目录
│   ├── 01_KNN_Explained.md          # KNN详解
│   ├── 02_KNN_vs_Traditional_ML.md  # KNN vs 传统机器学习
│   ├── 03_Algorithm_Comparison.md   # 算法全面对比
│   ├── 04_CNN_Explained.md          # CNN生活化解释
│   └── 05_Real_World_ML.md          # 机器学习的真实工作
│
├── models/                           # 训练好的模型
│   ├── knn_model.pkl                # KNN模型
│   ├── cnn_model.h5                 # CNN模型（8×8）
│   └── mnist_cnn_best.h5            # 🆕 高精度CNN模型（28×28，99.68%准确率）
│
├── test_cnn_images/                  # 测试图片目录
│   └── *.png                        # 你的手写数字图片
│
└── outputs/                          # 输出结果
    ├── visualizations/               # 可视化图表
    │   ├── 01_algorithm_comparison.png   # 算法性能对比（8×8）
    │   ├── 02_cnn_training.png           # 基础CNN训练曲线（8×8）
    │   ├── 03_cnn_predictions.png        # 基础CNN预测结果（8×8）
    │   ├── 04_cnn_filters.png            # 基础CNN卷积核（8×8）
    │   ├── 05_different_filters.png      # 不同模型卷积核对比
    │   ├── 06_filter_differences.png     # 卷积核差异统计
    │   ├── 07_predictions_comparison.png # 多算法预测对比
    │   ├── 08_mnist_samples.png     # 🆕 MNIST数据集样本（28×28）
    │   ├── 09_data_augmentation.png # 🆕 数据增强效果
    │   ├── 10_confusion_matrix.png  # 🆕 混淆矩阵（训练时生成）
    │   ├── 11_training_history.png  # 🆕 高精度CNN训练曲线（训练时生成）
    │   ├── 12_predictions.png       # 🆕 高精度CNN预测结果（训练时生成）
    │   ├── 13_cnn_filters.png       # 🆕 高精度CNN卷积核（训练时生成）
    │   ├── 14_feature_maps.png      # 🆕 特征图可视化（训练时生成）
    │   └── 15_test_results.png      # 🆕 你的手写数字测试结果
    └── reports/                      # 分析报告
        └── training_history.pkl      # 🆕 训练历史数据
```

### 🚀 快速开始

#### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

#### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 运行示例

```bash
# 入门：KNN基础训练
python src/01_knn_basic.py

# 进阶：多算法对比
python src/04_compare_algorithms.py

# 高级：CNN实现（8×8数据集，快速体验）
python src/05_cnn_advanced.py

# 🆕 专业级：高精度CNN（28×28 MNIST，99.68%准确率）
python src/06_cnn_mnist_advanced.py

# 🆕 测试你的手写数字图片
python test_cnn_images.py --images-dir test_cnn_images
```

### 📚 学习路线

#### 初学者路线（3-4小时）

1. **理解KNN基础**（30分钟）
   ```bash
   python src/01_knn_basic.py
   ```
   阅读：`docs/01_KNN_Explained.md`

2. **理解K值的影响**（20分钟）
   ```bash
   python src/02_knn_tuning.py
   ```

3. **学习优化方法**（30分钟）
   ```bash
   python src/03_knn_optimization.py
   ```

4. **阅读文档**（1-2小时）
   - `docs/01_KNN_Explained.md`
   - `docs/02_KNN_vs_Traditional_ML.md`

#### 进阶路线（2-3小时）

1. **多算法对比**（30分钟）
   ```bash
   python src/04_compare_algorithms.py
   ```

2. **安装深度学习库**
   ```bash
   pip install tensorflow
   ```

3. **体验CNN**（30分钟）
   ```bash
   python src/05_cnn_advanced.py
   ```

4. **深入阅读**（1-2小时）
   - `docs/03_Algorithm_Comparison.md`
   - `docs/04_CNN_Explained.md`

#### 🆕 高级路线（专业级CNN）（2-3小时）

1. **训练高精度模型**（20分钟，需要等待训练完成）
   ```bash
   python src/06_cnn_mnist_advanced.py
   ```
   
   这将训练一个99.68%准确率的专业级CNN模型，包含：
   - ✅ 完整MNIST数据集（60,000训练样本，28×28图片）
   - ✅ 深层CNN架构（3组卷积块）
   - ✅ 批归一化（BatchNormalization）
   - ✅ 数据增强（旋转、平移、缩放）
   - ✅ 学习率衰减
   - ✅ 早停机制
   - ✅ 自动保存最佳模型

2. **测试你自己的手写数字**（10分钟）
   
   准备测试图片：
   - 在纸上写几个数字，拍照或截图
   - 将图片放到 `test_cnn_images/` 目录
   - 图片可以是白底黑字（脚本会自动反色处理）
   
   运行测试：
   ```bash
   python test_cnn_images.py
   ```
   
   脚本会自动：
   - 🔄 将白底黑字转换为黑底白字
   - 📏 调整图片大小为28×28
   - 🎯 使用高精度模型预测
   - 📊 显示每张图片的预测结果和置信度
   - 🎨 生成可视化结果图

3. **理解优化技巧**（30分钟）
   
   查看训练输出和可视化结果：
   - `outputs/visualizations/08_mnist_samples.png` - MNIST数据集样本
   - `outputs/visualizations/09_data_augmentation.png` - 数据增强效果
   - `outputs/visualizations/10_confusion_matrix.png` - 混淆矩阵
   - `outputs/visualizations/11_training_history.png` - 训练曲线
   - `outputs/visualizations/12_predictions.png` - 预测结果
   - `outputs/visualizations/13_cnn_filters.png` - 学到的卷积核
   - `outputs/visualizations/14_feature_maps.png` - 特征图可视化

4. **深入研究**（1小时）
   - 阅读代码中的注释，理解每个优化技巧
   - 尝试修改参数（学习率、batch size、网络深度）
   - 对比不同配置的效果

### 🎓 核心概念

#### KNN (K-Nearest Neighbors)
- **原理**：找最相似的K个样本，投票决定
- **优点**：简单易懂，无需训练
- **缺点**：预测慢，占内存大
- **适用**：教学、小数据集

#### CNN (Convolutional Neural Network)
- **原理**：卷积层自动学习层次化特征
- **优点**：准确率最高，适合图像
- **缺点**：训练时间长，需要数据
- **适用**：图像识别、计算机视觉

### 🆕 高精度CNN模型详解

#### 模型特点

**输入格式**
- 图片尺寸：28×28像素
- 颜色：灰度图（单通道）
- 数值范围：0.0 - 1.0（归一化）
- 格式：黑底白字（MNIST标准）

**模型架构**
```
输入层 (28×28×1)
    ↓
第一组卷积块:
  - Conv2D(32) + BatchNorm + Conv2D(32) + BatchNorm
  - MaxPooling2D + Dropout(0.25)
    ↓
第二组卷积块:
  - Conv2D(64) + BatchNorm + Conv2D(64) + BatchNorm
  - MaxPooling2D + Dropout(0.25)
    ↓
第三组卷积块:
  - Conv2D(128) + BatchNorm
  - MaxPooling2D + Dropout(0.4)
    ↓
全连接层:
  - Dense(256) + BatchNorm + Dropout(0.5)
  - Dense(128) + Dropout(0.5)
    ↓
输出层 (10类，softmax)
```

**应用的优化技巧**
1. **批归一化（BatchNormalization）**
   - 加速训练收敛
   - 提高模型稳定性
   
2. **Dropout正则化**
   - 防止过拟合
   - 提高泛化能力

3. **数据增强（Data Augmentation）**
   - 随机旋转：±10°
   - 随机平移：±10%
   - 随机缩放：±10%
   - 剪切变换：±10%

4. **学习率衰减**
   - 初始学习率：0.001
   - 自动降低50%（5个epoch无改善）
   - 最低学习率：1e-7

5. **早停机制**
   - 监控验证集损失
   - 15个epoch无改善则停止
   - 自动恢复最佳权重

**训练结果**
- 测试集准确率：**99.68%**
- 训练时间：约17分钟（M系列芯片）
- 模型大小：约2.5 MB
- 训练轮数：59 epochs（早停）

### 📊 数据集

#### 基础数据集（scikit-learn digits）
- **图片数量**：1,797张
- **图片大小**：8×8像素
- **类别**：0-9共10个数字
- **特点**：适合快速实验和学习
- **使用脚本**：`01-05`系列脚本

#### 🆕 完整MNIST数据集
- **图片数量**：70,000张（60,000训练 + 10,000测试）
- **图片大小**：28×28像素
- **类别**：0-9共10个数字
- **特点**：业界标准数据集，专业级应用
- **使用脚本**：`src/06_cnn_mnist_advanced.py`

### 🎯 使用测试脚本

#### 测试你自己的手写数字

`test_cnn_images.py` 脚本可以让你用训练好的高精度模型测试自己的手写数字图片。

**基本用法**

```bash
# 使用默认设置（推荐）
python test_cnn_images.py

# 指定图片目录
python test_cnn_images.py --images-dir /path/to/your/images

# 使用其他模型
python test_cnn_images.py --model models/other_model.h5

# 如果图片已经是黑底白字（不需要反色）
python test_cnn_images.py --no-invert
```

**准备测试图片**

1. 在白纸上写几个数字（0-9）
2. 用手机拍照或电脑截图
3. 将图片保存到 `test_cnn_images/` 目录
4. 运行测试脚本

**图片要求**
- ✅ 支持格式：PNG, JPG, JPEG, BMP
- ✅ 白底黑字（推荐）或黑底白字
- ✅ 尽量一张图片一个数字
- ✅ 数字清晰可见
- ⚠️ 背景简洁，避免复杂背景

**输出结果**

脚本会输出：
1. **结果表格**：文件名、预测数字、置信度
2. **可视化图片**：`outputs/visualizations/15_test_results.png`
3. **准确率统计**：如果文件名包含数字标签

**示例输出**

```
================================================================================
🚀 高精度手写数字识别测试（MNIST 28×28 模型）
================================================================================
📁 测试图片目录: test_cnn_images
📊 找到 5 张图片
✅ 正在加载模型: models/mnist_cnn_best.h5
   模型输入形状: (None, 28, 28, 1)

================================================================================
📊 预测结果
================================================================================
文件名                                    | 预测数字 | 置信度
------------------------------------------------------------------------
digit_3.png                               | 3        | 99.85%
digit_7.png                               | 7        | 98.32%
my_handwriting_5.png                      | 5        | 99.12%

✅ 测试完成，共处理 5 张图片。
📈 准确率: 100.00%
✅ 可视化结果已保存: outputs/visualizations/15_test_results.png
```

### 🔧 依赖项

核心依赖：
- `numpy` - 数值计算
- `matplotlib` - 可视化
- `scikit-learn` - 机器学习算法
- `joblib` - 模型保存
- `pillow` - 图片处理

深度学习依赖（CNN）：
- `tensorflow` - 深度学习框架（推荐2.x版本）

安装所有依赖：
```bash
pip install -r requirements.txt
```

### 💡 常见问题

#### Q1: 训练高精度CNN模型时报错缺少seaborn？
A: 已经修复，不再需要seaborn。如果使用旧版本代码，可以：
```bash
pip install seaborn
```

#### Q2: 测试图片时预测结果不准确？
A: 确保：
- 图片清晰，数字完整
- 背景干净，避免复杂背景
- 使用 `--no-invert` 如果图片已经是黑底白字
- 一张图片只有一个数字

#### Q3: 训练CNN模型需要多长时间？
A: 
- 基础CNN（8×8）：约1分钟
- 高精度CNN（28×28）：约15-20分钟（取决于硬件）
- GPU加速可以显著减少训练时间

#### Q4: 如何使用训练好的模型？
A: 
```python
from tensorflow import keras
import numpy as np
from PIL import Image

# 加载模型
model = keras.models.load_model('models/mnist_cnn_best.h5')

# 加载并预处理图片
img = Image.open('your_image.png').convert('L')
img = img.resize((28, 28))
arr = np.asarray(img, dtype=np.float32) / 255.0
arr = arr.reshape(1, 28, 28, 1)

# 预测
prediction = model.predict(arr)
digit = np.argmax(prediction)
confidence = prediction[0][digit]

print(f'预测数字: {digit}, 置信度: {confidence*100:.2f}%')
```

#### Q5: 模型文件在哪里？
A: 
- 基础模型：`models/cnn_model.h5`（8×8）
- 高精度模型：`models/mnist_cnn_best.h5`（28×28，99.68%准确率）
- 训练后会自动保存到models目录

### 🎨 可视化结果

训练和测试过程会生成多个可视化图表，帮助理解模型：

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 算法对比 | `01_algorithm_comparison.png` | KNN/SVM/RF/MLP算法性能对比（8×8数据集） |
| CNN训练曲线 | `02_cnn_training.png` | 基础CNN训练过程（准确率和损失） |
| CNN预测结果 | `03_cnn_predictions.png` | 基础CNN预测20个样本示例（8×8） |
| CNN卷积核 | `04_cnn_filters.png` | 基础CNN第一层学到的32个卷积核 |
| 不同模型卷积核 | `05_different_filters.png` | 三个不同模型学到的卷积核对比 |
| 卷积核差异对比 | `06_filter_differences.png` | 三个模型的32个卷积核差异统计 |
| 多算法预测对比 | `07_predictions_comparison.png` | KNN/SVM/RF/MLP四种算法预测结果对比 |
| **MNIST样本** | `08_mnist_samples.png` | **28×28高分辨率MNIST数据集样本** |
| **数据增强** | `09_data_augmentation.png` | **数据增强效果（旋转、平移、缩放）** |
| **测试结果** | `15_test_results.png` | **你的手写数字识别结果（高精度CNN）** |

注：粗体标记的是高精度CNN（28×28）相关的可视化结果。其他图表来自基础训练脚本（8×8数据集）。

高精度CNN训练完成后，还会生成以下图表（保存在 `outputs/visualizations/` 目录）：
- `10_confusion_matrix.png` - 混淆矩阵（各数字识别准确率详情）
- `11_training_history.png` - 高精度CNN训练曲线（准确率和损失）
- `12_predictions.png` - 高精度CNN预测30个样本示例（28×28）
- `13_cnn_filters.png` - 高精度CNN学到的32个卷积核
- `14_feature_maps.png` - 高精度CNN特征图可视化

### 📖 详细文档

所有文档位于 `docs/` 目录：

1. **KNN详解** - KNN算法原理、标签必要性、K值选择
2. **KNN vs 传统机器学习** - 懒惰学习的本质
3. **算法全面对比** - 5种算法详细对比和选择指南
4. **CNN生活化解释** - 用生活例子解释CNN每一步
5. **机器学习的真实工作** - 实际项目中的工作流程

### 🚀 快速开始指南

#### 完全新手（第一次接触机器学习）

```bash
# 1. 从最简单的KNN开始
python src/01_knn_basic.py

# 2. 阅读KNN详解
cat docs/01_KNN_Explained.md

# 3. 了解不同算法的对比
python src/04_compare_algorithms.py
```

#### 有基础想深入学习

```bash
# 1. 安装深度学习库
pip install tensorflow

# 2. 训练高精度CNN模型（会花20分钟）
python src/06_cnn_mnist_advanced.py

# 3. 测试你自己的手写数字
# 先准备几张手写数字图片放到 test_cnn_images/ 目录
python test_cnn_images.py
```

#### 只想快速测试模型

如果已经有训练好的模型：

```bash
# 直接测试你的手写数字
python test_cnn_images.py --model models/mnist_cnn_best.h5
```

### 📈 项目亮点

#### ✨ 教学友好
- 代码注释详细，每行都有中文说明
- 循序渐进，从简单的KNN到复杂的CNN
- 包含5份详细的技术文档
- 所有结果都有可视化展示

#### 🎯 实用性强
- 提供训练好的高精度模型（99.68%准确率）
- 包含完整的测试脚本
- 可以直接识别你自己的手写数字
- 代码可以直接用于实际项目

#### 🔬 技术深度
- 涵盖传统机器学习到深度学习
- 应用多种优化技巧（数据增强、学习率衰减、早停等）
- 详细的性能对比和分析
- 可视化模型内部学到的特征

#### 📊 完整性
- 从数据加载到模型部署的完整流程
- 包含训练、验证、测试、预测所有环节
- 提供多个数据集（8×8和28×28）
- 8个可视化图表全面展示结果

### 🤝 贡献

欢迎提交Issue和Pull Request！

### 📄 许可证

MIT License

### 🌟 致谢

- scikit-learn团队提供的优秀工具
- TensorFlow/Keras团队
- 所有贡献者

---

## English Version

A comprehensive machine learning tutorial project, learning various algorithms from traditional ML (KNN) to deep learning (CNN) through handwritten digit recognition.

### 📊 Project Overview

This project implements 5 different ML algorithms for digit recognition:
- **KNN** - Simplest algorithm for beginners
- **SVM** - Efficient classifier
- **Random Forest** - Ensemble learning
- **MLP** - Traditional neural network
- **CNN** - Best for image recognition ⭐

### 🎯 Performance Comparison

| Algorithm | Dataset | Accuracy | Training Time | Use Case |
|-----------|---------|----------|---------------|----------|
| KNN | 8×8 | ~92% | 0.001s | Education |
| SVM | 8×8 | ~97% | 0.35s | Production |
| Random Forest | 8×8 | ~95% | 0.12s | General |
| MLP | 8×8 | ~98% | 12s | General DL |
| CNN (Basic) | 8×8 | ~98% | 45s | Image Recognition |
| **CNN (Advanced)** | **28×28 MNIST** | **99.68%** | **17min** | **Professional** ⭐ |

### 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/handwritten-digit-recognition.git
cd handwritten-digit-recognition

# Install dependencies
pip install -r requirements.txt

# Run basic KNN
python src/01_knn_basic.py

# Compare all algorithms
python src/04_compare_algorithms.py

# Run CNN (requires TensorFlow)
pip install tensorflow
python src/05_cnn_advanced.py

# 🆕 Train high-accuracy CNN (99.68% accuracy)
python src/06_cnn_mnist_advanced.py

# 🆕 Test with your own handwritten digits
python test_cnn_images.py
```

### 📚 Learning Path

1. **Beginner**: Start with KNN basics (`src/01_knn_basic.py`)
2. **Intermediate**: Compare multiple algorithms (`src/04_compare_algorithms.py`)
3. **Advanced**: Implement CNN (`src/05_cnn_advanced.py`)
4. **🆕 Professional**: Train high-accuracy CNN (`src/06_cnn_mnist_advanced.py`)
5. **🆕 Practice**: Test your own handwritten digits (`test_cnn_images.py`)

### 🆕 New Features

#### High-Accuracy CNN Model
- **99.68% accuracy** on MNIST test set
- Full MNIST dataset (60,000 training images, 28×28)
- Advanced techniques: BatchNormalization, Data Augmentation, Learning Rate Decay, Early Stopping
- Comprehensive visualizations (8 different charts)
- Training time: ~17 minutes

#### Test Script for Your Own Images
- Automatic preprocessing (invert colors, resize to 28×28, normalize)
- Batch prediction with confidence scores
- Visualization of results
- Supports PNG, JPG, JPEG, BMP formats

**Usage:**
```bash
# Test images in default directory
python test_cnn_images.py

# Specify custom directory
python test_cnn_images.py --images-dir /path/to/images

# Use different model
python test_cnn_images.py --model models/other_model.h5
```

### 📖 Documentation

All documentation is available in the `docs/` directory:

1. **KNN Explained** - Algorithm principles, label importance, K value selection
2. **KNN vs Traditional ML** - Understanding lazy learning
3. **Algorithm Comparison** - Comprehensive comparison and selection guide
4. **CNN Explained** - CNN concepts with everyday examples
5. **Real World ML** - Actual project workflows

### 🎯 Model Comparison

#### Basic CNN vs High-Accuracy CNN

| Feature | Basic CNN (8×8) | High-Accuracy CNN (28×28) |
|---------|-----------------|---------------------------|
| Dataset | sklearn digits (1,797) | MNIST (60,000) |
| Image Size | 8×8 pixels | 28×28 pixels |
| Accuracy | ~98% | **99.68%** |
| Training Time | ~1 min | ~17 min |
| Model Size | ~500 KB | ~2.5 MB |
| Optimization | Basic | Advanced (BatchNorm, Data Aug, etc.) |
| Use Case | Learning | Professional Applications |

#### Optimization Techniques

| Technique | Basic CNN | High-Accuracy CNN |
|-----------|-----------|-------------------|
| Data Augmentation | ❌ | ✅ (Rotation, Translation, Zoom) |
| Batch Normalization | ❌ | ✅ |
| Learning Rate Decay | ❌ | ✅ |
| Early Stopping | ❌ | ✅ |
| Multiple Conv Blocks | ✅ (2 blocks) | ✅ (3 blocks) |
| Dropout | ✅ | ✅ (Multiple layers) |

### 🌟 Project Highlights

- ✅ **Beginner-Friendly**: Detailed Chinese comments, step-by-step tutorials
- ✅ **Practical**: Pre-trained high-accuracy model included
- ✅ **Complete**: From data loading to deployment
- ✅ **Visual**: 8+ visualization charts
- ✅ **Production-Ready**: Test script for real-world use

### 🤝 Contributing

Issues and Pull Requests are welcome!

### 📄 License

MIT License

---

**Made with ❤️ for ML learners**

