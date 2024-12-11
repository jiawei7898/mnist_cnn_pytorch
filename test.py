"""
checkpoints/test_log.txt
    文件内容：这是主日志文件，记录了程序运行过程中的所有重要事件和信息。
    用途：
    记录程序启动时间、设备信息（CPU/GPU）、数据加载情况、模型加载情况、测试过程中的每一步操作（如批次处理、损失计算、准确率计算等）。
    记录每个批次的耗时、损失值、准确率等详细信息。
    记录任何异常或错误信息，帮助调试和排查问题。

checkpoints/test_performance.txt
    文件内容：记录了整个测试过程的性能指标，包括测试时间、总损失、总准确率等。
    用途：
    提供一个简洁的总结，方便快速查看测试的整体表现。
    每次测试后，新的性能结果会被追加到文件末尾，形成历史记录。

checkpoints/classification_report.txt
    文件内容：保存了详细的分类报告，包括每个类别的精确度（Precision）、召回率（Recall）、F1 分数（F1-Score）以及支持度（Support）。
    用途：
    提供对模型分类性能的详细评估，特别是对于多类别分类任务，可以了解模型在不同类别上的表现。
    帮助识别哪些类别容易被误分类，从而指导模型的改进方向。

checkpoints/confusion_matrix.png
    文件内容：这是一个图像文件，展示了混淆矩阵（Confusion Matrix），用于直观地显示模型的分类结果。
    用途：
    混淆矩阵是评估分类模型的重要工具，它显示了模型在每个类别上的预测情况，特别是哪些类别被正确分类，哪些类别被误分类。
    通过可视化混淆矩阵，可以更直观地理解模型的分类行为，发现潜在的问题（如某些类别之间的混淆）。

checkpoints/test_metrics.png
    文件内容：这是一个图像文件，展示了测试过程中损失值和准确率的变化情况。
    用途：
    该图包含两个子图：
    左侧子图展示了测试损失（Test Loss），帮助评估模型的拟合程度。通常，较低的损失值表示模型在测试集上的表现较好。
    右侧子图展示了测试准确率（Test Accuracy），直接反映了模型在测试集上的分类性能。
    通过这两个图表，可以直观地了解模型的性能，并与训练过程中的损失和准确率进行对比，评估模型是否过拟合或欠拟合。
"""

import torch
import torch.nn as nn
from models.cnn import CNN
from utils.dataset import load_data
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
import os
import sys  # 添加 sys 模块

# 配置日志记录
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('logs/test_log.txt', mode='a'),
        logging.StreamHandler(sys.stdout)  # 将日志输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


# 加载数据
def load_test_data(batch_size=64):
    """加载测试数据集"""
    try:
        _, test_loader = load_data(batch_size)
        logger.info(f"Loaded test data with batch size {batch_size}.")
        return test_loader
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise


# 实例化模型并加载权重
def load_model(checkpoint_path):
    """加载预训练模型"""
    try:
        # 检查检查点文件是否存在
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        model = CNN().to(device)

        # 使用 weights_only=True 来提高安全性
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # 只加载模型的参数部分
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"Model loaded from {checkpoint_path}.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# 测试模型
def test_model(model, test_loader, criterion):
    """测试模型并返回性能指标"""
    test_loss = 0
    correct = 0
    all_labels = []
    all_preds = []

    logger.info(f"{'=' * 50}\nStarting Testing...\n{'=' * 50}")
    start_time = time.time()

    try:
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                batch_start_time = time.time()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # 记录所有标签和预测值
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time

                # 每10个批次输出一次
                if (batch_idx + 1) % 10 == 0:
                    batch_accuracy = (predicted == labels).sum().item() / labels.size(0) * 100
                    logger.info(
                        f'Batch [{batch_idx + 1}/{len(test_loader)}], Loss: {loss.item():.4f}, '
                        f'Accuracy: {batch_accuracy:.2f}%, Batch Time: {batch_duration:.2f}s'
                    )

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Testing completed in {total_time:.2f} seconds.\n{'=' * 50}")

        accuracy = correct / len(test_loader.dataset)
        test_loss /= len(test_loader)
        logger.info(f'Total Test Loss: {test_loss:.4f}, Total Accuracy: {accuracy:.2f}%')

        return test_loss, accuracy, all_labels, all_preds, total_time
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")
        raise


# 保存性能结果
def save_performance_results(test_loss, accuracy, total_time):
    """保存测试性能结果"""
    try:
        performance = (
            f'\nTest Performance on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'Model Weight File: checkpoints/best_model_20241211_093451.pth\n'
            f'Test Loss: {test_loss:.4f}\n'
            f'Total Accuracy: {accuracy:.2f}%\n'
            f'Testing Time: {time.strftime("%H:%M:%S", time.gmtime(total_time))}\n'
        )
        logger.info(performance)
        with open('logs/test_performance.txt', 'a') as f:
            f.write(performance)
    except Exception as e:
        logger.error(f"Failed to save performance results: {e}")
        raise


# 保存分类报告
def save_classification_report(all_labels, all_preds):
    """保存分类报告"""
    try:
        classification_rep = classification_report(all_labels, all_preds, digits=4)
        logger.info("Classification Report:\n" + classification_rep)
        with open('logs/classification_report.txt', 'w') as f:
            f.write(classification_rep)
    except Exception as e:
        logger.error(f"Failed to save classification report: {e}")
        raise


# 绘制混淆矩阵
def plot_confusion_matrix(all_labels, all_preds):
    """绘制并保存混淆矩阵"""
    try:
        conf_matrix = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('checkpoints/confusion_matrix.png')
        plt.show()
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")
        raise


# 绘制测试损失和准确率图
def plot_test_metrics(test_loss, accuracy):
    """绘制并保存测试损失和准确率图"""
    try:
        plt.figure(figsize=(12, 5))

        # 绘制测试损失
        plt.subplot(1, 2, 1)
        plt.plot([test_loss], marker='o', label='Test Loss')
        plt.xlabel('Test')
        plt.ylabel('Loss')
        plt.title('Test Loss')
        plt.legend()

        # 绘制测试准确率
        plt.subplot(1, 2, 2)
        plt.plot([accuracy], marker='o', label='Test Accuracy', color='orange')
        plt.xlabel('Test')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('checkpoints/test_metrics.png')
        plt.show()
    except Exception as e:
        logger.error(f"Failed to plot test metrics: {e}")
        raise


# 主函数
def main():
    try:
        # 创建检查点目录
        os.makedirs('checkpoints', exist_ok=True)
        logger.info("Checkpoints directory created or already exists.")

        # 加载测试数据
        test_loader = load_test_data(batch_size=64)

        # 加载模型
        checkpoint_path = 'checkpoints/best_model_20241211_093451.pth'
        model = load_model(checkpoint_path)

        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 测试模型并获取性能指标
        test_loss, accuracy, all_labels, all_preds, total_time = test_model(model, test_loader, criterion)

        # 保存性能结果
        save_performance_results(test_loss, accuracy, total_time)

        # 保存分类报告
        save_classification_report(all_labels, all_preds)

        # 绘制混淆矩阵
        plot_confusion_matrix(all_labels, all_preds)

        # 绘制测试损失和准确率图
        plot_test_metrics(test_loss, accuracy)

        logger.info("All processes completed successfully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)  # 退出程序，返回非零状态码表示失败


if __name__ == "__main__":
    main()
