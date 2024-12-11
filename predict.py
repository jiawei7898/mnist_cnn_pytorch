import torch
from models.cnn import CNN
from PIL import Image
import time
import logging
import os
import sys
from torchvision import transforms


# 配置日志记录
def setup_logging(log_dir='logs', log_file='prediction_log.txt'):
    """设置日志记录，确保日志目录存在"""
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 构建日志文件的完整路径
    log_path = os.path.join(log_dir, log_file)

    # 配置日志格式和处理器
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(sys.stdout)  # 将日志输出到控制台
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = setup_logging()
logger.info(f"Using device: {device}")


# 检查检查点文件是否存在
def check_checkpoint_exists(checkpoint_path):
    """检查模型权重文件是否存在"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    logger.info(f"Found checkpoint file: {checkpoint_path}")


# 实例化模型并加载权重
def load_model(checkpoint_path):
    """加载预训练模型"""
    try:
        check_checkpoint_exists(checkpoint_path)
        model = CNN().to(device)

        # 使用 weights_only=True 确保只加载模型权重
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # 检查 checkpoint 是否包含 'model_state_dict'，如果包含则从中提取模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果 checkpoint 是一个纯粹的 state_dict，则直接加载
            model.load_state_dict(checkpoint)

        model.eval()
        logger.info("Model loaded and set to evaluation mode.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


# 图像预处理
def preprocess_image(image_path):
    """加载图像并进行预处理"""
    try:
        transform = transforms.Compose([
            transforms.Grayscale(),  # 确保输入图像是灰度图
            transforms.Resize((28, 28)),  # 调整图像大小为 28x28
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize((0.1307,), (0.3081,))  # 标准化
        ])
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        image = transform(image).unsqueeze(0)  # 增加一个批次维度
        logger.info(f"Image preprocessed from: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        sys.exit(1)


# 预测函数
def predict(image_path, model):
    """使用模型进行预测"""
    try:
        start_time = time.time()  # 记录预测开始时间
        image = preprocess_image(image_path)
        image = image.to(device)
        with torch.no_grad():  # 在预测阶段不计算梯度
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
        predicted_digit = predicted.item()
        end_time = time.time()  # 记录预测结束时间
        prediction_time = end_time - start_time
        logger.info(f"Predicted digit: {predicted_digit}")
        logger.info(f"Time spent on prediction: {prediction_time:.4f} seconds")
        return predicted_digit, prediction_time
    except Exception as e:
        logger.error(f"Failed to predict: {e}")
        sys.exit(1)


# 可视化预测结果
def visualize_prediction(image_path, predicted_digit):
    """显示原始图像和预测结果"""
    try:
        import matplotlib.pyplot as plt
        image = Image.open(image_path).convert('L')
        plt.imshow(image, cmap='gray')
        plt.title(f'Predicted Digit: {predicted_digit}')
        plt.axis('off')
        plt.show()
    except Exception as e:
        logger.error(f"Failed to visualize prediction: {e}")


# 主函数
def main():
    try:
        # 加载模型
        checkpoint_path = 'checkpoints/best_model_20241211_111651.pth'  # 模型权重文件路径
        model = load_model(checkpoint_path)

        # 配置图像路径
        image_path = './data/pre/07.png'  # 替换为你的图像路径
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            sys.exit(1)

        # 进行预测
        predicted_digit, prediction_time = predict(image_path, model)

        # 可视化预测结果
        visualize_prediction(image_path, predicted_digit)

        logger.info("Prediction completed successfully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()