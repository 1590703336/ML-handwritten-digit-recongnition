#!/usr/bin/env python3
"""
批量验证高精度 CNN 手写数字模型（MNIST 28×28）。

默认会加载 `models/mnist_cnn_best.h5`（99.68%准确率的高精度模型），
并读取指定目录下的图片文件，逐张送入模型并打印预测结果及置信度。
处理白底黑字图片，转换为28×28黑底白字格式。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - 依赖缺失时直接退出
    raise SystemExit("❌ 未安装 TensorFlow，请先 `pip install tensorflow` 后再运行本脚本。") from exc

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mnist_cnn_best.h5"
)
DEFAULT_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Pillow<9.1
    RESAMPLE = Image.LANCZOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量测试高精度CNN模型（28×28 MNIST）对本地手写数字图片的识别效果")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="keras 模型文件路径（默认使用 mnist_cnn_best.h5，准确率99.68%%）",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/Users/huanzhang/code/ML/handwritten-digit-recognition/test_cnn_images"),
        help="待测试图片所在目录",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=255.0,
        help="像素归一化时使用的最大值（MNIST标准为255.0）",
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="不反色（默认会反色，因为白底黑字需要转换为黑底白字）",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=None,
        help="要纳入测试的图片扩展名（不区分大小写），例如: --ext .png .jpg",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="可视化显示所有图片和预测结果",
    )
    return parser.parse_args()


def collect_images(directory: Path, extensions: Sequence[str]) -> List[Path]:
    exts = tuple(ext.lower() for ext in extensions)
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in exts
    ]
    return files


def ensure_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise SystemExit(f"❌ 模型文件不存在：{model_path}")
    print(f"✅ 正在加载模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape
    if len(input_shape) != 4:
        raise SystemExit(f"⚠️ 暂不支持输入形状 {input_shape} 的模型（期望为 N,H,W,C）")
    print(f"   模型输入形状: {input_shape}")
    return model


def preprocess_image(
    image_path: Path,
    target_hw: Tuple[int, int],
    channels: int,
    scale_max: float,
    invert: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    预处理图片：
    1. 转为灰度图
    2. 反色（白底黑字 -> 黑底白字，MNIST标准格式）
    3. 调整为28×28
    4. 归一化到0-1
    
    返回: (预处理后的数组, 原始图片数组用于可视化)
    """
    with Image.open(image_path) as img:
        # 保存原始图片用于可视化
        original = img.copy()
        
        if channels == 1:
            img = img.convert("L")
            # 白底黑字 -> 黑底白字（MNIST标准）
            if invert:
                img = ImageOps.invert(img)
        else:
            img = img.convert("RGB")
        
        # 调整大小为28×28（MNIST标准）
        img = img.resize(target_hw[::-1], RESAMPLE)  # Pillow 使用 (W,H)
        arr = np.asarray(img, dtype=np.float32)
    
    if channels == 1:
        arr = arr[..., np.newaxis]
    else:
        arr = arr[..., :channels]
    
    # 归一化到0-1
    max_value = scale_max if scale_max else float(arr.max() or 1.0)
    arr /= max_value
    
    # 转换原始图片为数组
    original_arr = np.asarray(original.convert("L"), dtype=np.uint8)
    
    return arr, original_arr


def batch_predict(
    model: tf.keras.Model, tensors: Iterable[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    batch = np.stack(list(tensors), axis=0)
    predictions = model.predict(batch, verbose=0)
    labels = np.argmax(predictions, axis=1)
    confidences = predictions[np.arange(len(predictions)), labels]
    return labels, confidences


def format_row(values: Sequence[str], widths: Sequence[int]) -> str:
    return " | ".join(text.ljust(width) for text, width in zip(values, widths))


def main() -> None:
    args = parse_args()
    image_exts = args.ext if args.ext else DEFAULT_IMAGE_EXTS
    images = collect_images(args.images_dir, image_exts)
    if not images:
        raise SystemExit(
            f"⚠️ 目录 {args.images_dir} 内未找到图片（支持扩展名: {', '.join(image_exts)}）"
        )

    print("="*80)
    print("🚀 高精度手写数字识别测试（MNIST 28×28 模型）")
    print("="*80)
    print(f"📁 测试图片目录: {args.images_dir}")
    print(f"📊 找到 {len(images)} 张图片")
    
    model = ensure_model(args.model)
    _, height, width, channels = model.input_shape
    height = int(height or 0)
    width = int(width or 0)
    channels = int(channels or 1)
    
    if not height or not width:
        raise SystemExit(f"⚠️ 无法从模型输入形状 {model.input_shape} 推断尺寸")
    
    print(f"🔧 图片预处理设置:")
    print(f"   - 目标尺寸: {height}×{width}")
    print(f"   - 反色处理: {'否' if args.no_invert else '是（白底黑字→黑底白字）'}")
    print(f"   - 归一化: 0-{args.scale_max}")
    
    # 预处理所有图片
    invert = not args.no_invert  # 默认进行反色
    processed_data = [
        preprocess_image(img_path, (height, width), channels, args.scale_max, invert)
        for img_path in images
    ]
    tensors = [data[0] for data in processed_data]
    originals = [data[1] for data in processed_data]

    # 批量预测
    print("\n🧠 正在进行预测...")
    preds, confidences = batch_predict(model, tensors)
    
    # 打印结果表格
    print("\n" + "="*80)
    print("📊 预测结果")
    print("="*80)
    header = ["文件名", "预测数字", "置信度"]
    widths = [max(len(header[0]), 40), len(header[1])+2, len(header[2])+2]
    print(format_row(header, widths))
    print("-" * (sum(widths) + 6))

    for img_path, label, score in zip(images, preds, confidences):
        name = img_path.name[: widths[0]]
        row = [
            name,
            str(int(label)),
            f"{score * 100:5.2f}%",
        ]
        print(format_row(row, widths))

    print("\n✅ 测试完成，共处理 {} 张图片。".format(len(images)))
    
    # 统计准确率（如果文件名包含数字标签）
    correct = 0
    labeled = 0
    for img_path, label in zip(images, preds):
        # 尝试从文件名中提取数字
        for char in img_path.stem:
            if char.isdigit():
                true_label = int(char)
                labeled += 1
                if true_label == label:
                    correct += 1
                break
    
    if labeled > 0:
        acc = correct / labeled * 100
        print(f"\n📈 在 {labeled} 张有标签的图片中，准确识别 {correct} 张，准确率: {acc:.2f}%")
    
    # 可视化结果
    if args.visualize or True:  # 总是显示可视化
        print("\n🎨 生成可视化结果...")
        n_images = len(images)
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3.5))
        if n_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        fig.suptitle('手写数字识别结果（高精度CNN - 99.68%准确率）', 
                     fontsize=16, fontweight='bold')
        
        for i, (img_path, original, label, score) in enumerate(zip(images, originals, preds, confidences)):
            ax = axes[i]
            ax.imshow(original, cmap='gray')
            
            # 显示文件名（缩短）
            filename = img_path.name
            if len(filename) > 20:
                filename = filename[:17] + "..."
            
            title = f'{filename}\n预测: {label} (置信度: {score*100:.1f}%)'
            
            # 如果置信度很高，用绿色；中等用黄色；低用红色
            if score >= 0.95:
                color = 'green'
            elif score >= 0.80:
                color = 'orange'
            else:
                color = 'red'
            
            ax.set_title(title, fontsize=9, color=color, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        output_path = Path('outputs/visualizations/15_test_results.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化结果已保存: {output_path}")
        plt.show()
        
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n用户中断。")

