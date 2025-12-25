import joblib
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def predict_my_image(image_path):
    # 1. 加载模型
    model = joblib.load('digit_model.pkl')
    
    # 2. 图片预处理 (Preprocessing Pipeline)
    # 打开图片并转为灰度图 (L)
    img = Image.open(image_path).convert('L')
    
    # ⚠️ 关键步骤：反转颜色
    # 训练集是黑底白字(数字是高数值)，你画的是白底黑字。
    # 所以必须把颜色反转过来！
    img = ImageOps.invert(img) 
    
    # 调整大小为 8x8 像素 (和训练集一致)
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    
    # 转为 numpy 数组
    img_array = np.array(img)
    
    # 数值归一化：把 0-255 的像素值压缩到 0-16 (和训练集一致)
    img_array = (img_array / 255.0) * 16.0
    
    # 拉平成 1x64 的向量
    input_vector = img_array.reshape(1, -1)
    
    # 3. 预测
    prediction = model.predict(input_vector)
    
    # 4. 展示结果
    plt.imshow(img_array, cmap='gray')
    plt.title(f"AI Predicts: {prediction[0]}")
    plt.axis('off')
    plt.show()
    
    print(f"🤖 AI 觉得你写的是: {prediction[0]}")

# 运行预测
try:
    predict_my_image('my_digit.png')
except FileNotFoundError:
    print("❌ 找不到图片！请先画一个 'my_digit.png' 放在目录下。")