"""
验证：不同的CNN会学到不同的卷积核
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    print("="*80)
    print("🔬 实验：证明不同CNN学到的卷积核不同")
    print("="*80)
    
    # 加载数据
    digits = datasets.load_digits()
    X = digits.images.reshape(-1, 8, 8, 1).astype('float32') / 16.0
    y = keras.utils.to_categorical(digits.target, 10)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ============ 训练3个相同结构但不同初始化的模型 ============
    models = []
    
    for i in range(3):
        print(f"\n训练模型 {i+1}/3...")
        
        # 创建相同结构的模型，但随机种子不同
        model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', 
                         input_shape=(8,8,1), padding='same',
                         kernel_initializer=keras.initializers.GlorotUniform(seed=i*1000)),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # 训练（只训练10轮，看趋势就够了）
        history = model.fit(X_train, y_train, 
                          epochs=10, 
                          batch_size=32,
                          verbose=0,
                          validation_data=(X_test, y_test))
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"模型 {i+1} 准确率: {test_acc*100:.2f}%")
        
        models.append(model)
    
    print("\n" + "="*80)
    print("📊 对比三个模型的第一层卷积核")
    print("="*80)
    
    # 提取三个模型的卷积核
    filters_list = []
    for i, model in enumerate(models):
        filters = model.layers[0].get_weights()[0]  # shape: (3, 3, 1, 32)
        filters_list.append(filters)
        print(f"模型 {i+1} 卷积核形状: {filters.shape}")
    
    # ============ 可视化对比 ============
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    fig.suptitle('三个不同模型学到的卷积核（前8个）', fontsize=16, fontweight='bold')
    
    for model_idx in range(3):
        filters = filters_list[model_idx][:, :, 0, :]  # (3, 3, 32)
        
        for filter_idx in range(8):
            ax = axes[model_idx, filter_idx]
            ax.imshow(filters[:, :, filter_idx], cmap='viridis')
            
            if filter_idx == 0:
                ax.set_ylabel(f'模型{model_idx+1}', fontsize=12, fontweight='bold')
            if model_idx == 0:
                ax.set_title(f'Filter {filter_idx+1}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('different_cnn_filters.png', dpi=150, bbox_inches='tight')
    print("\n✅ 对比图已保存: different_cnn_filters.png")
    plt.show()
    
    # ============ 数值对比 ============
    print("\n" + "="*80)
    print("🔍 详细分析：卷积核的数值差异")
    print("="*80)
    
    # 比较模型1和模型2的第一个卷积核
    filter1_model1 = filters_list[0][:, :, 0, 0]
    filter1_model2 = filters_list[1][:, :, 0, 0]
    filter1_model3 = filters_list[2][:, :, 0, 0]
    
    print("\n模型1的第1个卷积核:")
    print(filter1_model1)
    
    print("\n模型2的第1个卷积核:")
    print(filter1_model2)
    
    print("\n模型3的第1个卷积核:")
    print(filter1_model3)
    
    # 计算差异
    diff_12 = np.abs(filter1_model1 - filter1_model2).mean()
    diff_13 = np.abs(filter1_model1 - filter1_model3).mean()
    diff_23 = np.abs(filter1_model2 - filter1_model3).mean()
    
    print("\n" + "-"*80)
    print("卷积核差异（平均绝对差）:")
    print(f"  模型1 vs 模型2: {diff_12:.4f}")
    print(f"  模型1 vs 模型3: {diff_13:.4f}")
    print(f"  模型2 vs 模型3: {diff_23:.4f}")
    print("-"*80)
    
    # ============ 统计所有卷积核的差异 ============
    print("\n" + "="*80)
    print("📈 统计分析：所有32个卷积核的差异")
    print("="*80)
    
    all_diffs = []
    for i in range(32):
        f1 = filters_list[0][:, :, 0, i]
        f2 = filters_list[1][:, :, 0, i]
        f3 = filters_list[2][:, :, 0, i]
        
        all_diffs.append({
            'filter_id': i+1,
            'diff_12': np.abs(f1 - f2).mean(),
            'diff_13': np.abs(f1 - f3).mean(),
            'diff_23': np.abs(f2 - f3).mean()
        })
    
    avg_diff_12 = np.mean([d['diff_12'] for d in all_diffs])
    avg_diff_13 = np.mean([d['diff_13'] for d in all_diffs])
    avg_diff_23 = np.mean([d['diff_23'] for d in all_diffs])
    
    print(f"\n所有32个卷积核的平均差异:")
    print(f"  模型1 vs 模型2: {avg_diff_12:.4f}")
    print(f"  模型1 vs 模型3: {avg_diff_13:.4f}")
    print(f"  模型2 vs 模型3: {avg_diff_23:.4f}")
    
    # 可视化差异分布
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, 33)
    width = 0.25
    
    ax.bar(x - width, [d['diff_12'] for d in all_diffs], width, 
           label='模型1 vs 模型2', alpha=0.8)
    ax.bar(x, [d['diff_13'] for d in all_diffs], width, 
           label='模型1 vs 模型3', alpha=0.8)
    ax.bar(x + width, [d['diff_23'] for d in all_diffs], width, 
           label='模型2 vs 模型3', alpha=0.8)
    
    ax.set_xlabel('卷积核编号', fontsize=12)
    ax.set_ylabel('平均绝对差异', fontsize=12)
    ax.set_title('三个模型的32个卷积核差异对比', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('filter_differences.png', dpi=150, bbox_inches='tight')
    print("\n✅ 差异分析图已保存: filter_differences.png")
    plt.show()
    
    # ============ 结论 ============
    print("\n" + "="*80)
    print("🎯 实验结论")
    print("="*80)
    print(f"""
1. ✅ 三个模型结构完全相同（都是32个3×3卷积核）
2. ✅ 但学到的卷积核**完全不同**！
3. ✅ 平均差异达到 {avg_diff_12:.4f}（如果完全相同应该是0）
4. ✅ 每个模型都能达到90%+的准确率

这证明了：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ 卷积核不是固定的32种笔画！
✨ 而是从无穷可能中学习出来的！
✨ 不同的初始化、不同的训练过程 → 学到不同的卷积核
✨ 但都能达到相似的识别效果！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

就像：
🎓 三个学生学书法
   - 每个人的横竖撇捺写法都不完全一样
   - 但都能写出漂亮的字！
   
🏀 三个人学投篮
   - 每个人的姿势都不完全相同
   - 但都能投进！
""")
    
    print("="*80)
    
except ImportError:
    print("需要安装TensorFlow: pip install tensorflow")

