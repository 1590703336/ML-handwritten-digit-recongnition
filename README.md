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

| 算法 | 准确率 | 训练时间 | 预测速度 | 推荐场景 |
|------|--------|---------|---------|---------|
| KNN | ~92% | 0.001秒 | 慢 | 教学、理解基础 |
| SVM | ~97% | 0.35秒 | 快 | 生产环境 |
| Random Forest | ~95% | 0.12秒 | 快 | 通用任务 |
| MLP | ~98% | 12秒 | 快 | 通用深度学习 |
| CNN | **~99%** | 45秒 | 快 | **图像识别最佳** ⭐ |

### 📁 项目结构

```
handwritten-digit-recognition/
├── README.md                          # 项目主文档
├── requirements.txt                   # Python依赖
├── .gitignore                        # Git忽略文件
│
├── src/                              # 源代码目录
│   ├── 01_knn_basic.py              # KNN基础训练（入门必看）
│   ├── 02_knn_tuning.py             # KNN调参实验
│   ├── 03_knn_optimization.py       # KNN优化方法
│   ├── 04_compare_algorithms.py     # 多算法对比
│   ├── 05_cnn_advanced.py           # CNN实现
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
│   └── cnn_model.h5                 # CNN模型
│
└── outputs/                          # 输出结果
    ├── visualizations/               # 可视化图表
    │   ├── knn_predictions.png
    │   ├── algorithm_comparison.png
    │   ├── cnn_training.png
    │   ├── cnn_filters.png
    │   └── ...
    └── reports/                      # 分析报告
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

# 高级：CNN实现（需要TensorFlow）
python src/05_cnn_advanced.py
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

### 📊 数据集

使用scikit-learn内置的digits数据集：
- **图片数量**：1797张
- **图片大小**：8×8像素
- **类别**：0-9共10个数字
- **特点**：适合快速实验和学习

### 🔧 依赖项

核心依赖：
- `numpy` - 数值计算
- `matplotlib` - 可视化
- `scikit-learn` - 机器学习算法
- `joblib` - 模型保存

可选依赖（CNN）：
- `tensorflow` - 深度学习框架

### 📖 详细文档

所有文档位于 `docs/` 目录：

1. **KNN详解** - KNN算法原理、标签必要性、K值选择
2. **KNN vs 传统机器学习** - 懒惰学习的本质
3. **算法全面对比** - 5种算法详细对比和选择指南
4. **CNN生活化解释** - 用生活例子解释CNN每一步
5. **机器学习的真实工作** - 实际项目中的工作流程

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

| Algorithm | Accuracy | Training Time | Prediction Speed | Use Case |
|-----------|----------|---------------|------------------|----------|
| KNN | ~92% | 0.001s | Slow | Education |
| SVM | ~97% | 0.35s | Fast | Production |
| Random Forest | ~95% | 0.12s | Fast | General |
| MLP | ~98% | 12s | Fast | General DL |
| CNN | **~99%** | 45s | Fast | **Images** ⭐ |

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
```

### 📚 Learning Path

1. **Beginner**: Start with KNN basics
2. **Intermediate**: Compare multiple algorithms
3. **Advanced**: Implement CNN

### 📖 Documentation

All documentation is available in the `docs/` directory in both Chinese and English (planned).

### 🤝 Contributing

Issues and Pull Requests are welcome!

### 📄 License

MIT License

---

**Made with ❤️ for ML learners**

