import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import CNN
from utils.dataset import load_data
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os
import time  # 添加time模块的导入


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, test_loader, num_epochs, log_file, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.log_file = log_file
        self.device = device
        self.setup_logging()
        self.best_val_accuracy = 0.0
        self.start_time = None
        self.end_time = None
        self.epoch_losses = []
        self.val_epoch_accuracies = []
        self.best_model_state_dict = None  # 用于存储最佳模型的状态字典

    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            filemode='w',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, epoch, loss, accuracy, checkpoint_path, is_best=False, last_model=False):
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'accuracy': accuracy
            }, checkpoint_path)

            log_info = {
                'checkpoint_path': checkpoint_path,
                'epoch': epoch,
                'loss': f'{loss:.4f}',
                'accuracy': f'{accuracy:.2f}%',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            if is_best:
                log_info['type'] = 'Best Model'
                log_info['validation_accuracy'] = f'{self.best_val_accuracy:.2f}%'
            elif last_model:
                log_info['type'] = 'Last Model'
                log_info['total_training_time'] = f'{self.end_time - self.start_time:.2f} seconds'
            else:
                log_info['type'] = 'Checkpoint'

            self.logger.info(f'Saved {log_info["type"]} at {log_info["timestamp"]}:')
            for key, value in log_info.items():
                self.logger.info(f'  - {key}: {value}')

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def validate_model(self):
        self.model.eval()  # 设置模型为评估模式
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        accuracy = correct_predictions / total_predictions * 100
        self.model.train()  # 回到训练模式
        return accuracy

    def plot_training_results(self):
        epochs = range(1, self.num_epochs + 1)

        plt.figure(figsize=(12, 5))

        # 绘制训练损失
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.epoch_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()

        # 绘制验证准确率
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_epoch_accuracies, label='Validation Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy Over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig('checkpoints/training_results.png')
        plt.show()

    def train(self):
        self.model.train()  # 设置模型为训练模式
        self.start_time = time.time()  # 记录训练开始时间

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                if (i + 1) % 100 == 0:  # 每100个batch输出一次
                    self.logger.info(
                        f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = correct_predictions / total_predictions * 100
            self.epoch_losses.append(epoch_loss)
            self.logger.info(
                f'Epoch [{epoch + 1}/{self.num_epochs}] finished with loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

            # 验证模型
            val_accuracy = self.validate_model()
            self.val_epoch_accuracies.append(val_accuracy)
            self.logger.info(f'Validation Accuracy: {val_accuracy:.2f}%')

            # 更新最佳模型的状态字典
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_model_state_dict = self.model.state_dict()
                self.logger.info(f'New best validation accuracy: {self.best_val_accuracy:.2f}%')

        self.end_time = time.time()  # 记录训练结束时间
        self.logger.info(f'Total training time: {self.end_time - self.start_time:.2f} seconds')

        # 保存最佳模型
        if self.best_model_state_dict is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.best_model_path = f'checkpoints/best_model_{timestamp}.pth'
            self.model.load_state_dict(self.best_model_state_dict)  # 加载最佳模型的状态字典
            self.save_checkpoint(self.num_epochs, epoch_loss, epoch_accuracy, self.best_model_path, is_best=True)
            self.logger.info(f'Best model saved to {self.best_model_path} with validation accuracy: {self.best_val_accuracy:.2f}%')

        # 保存最后一次的模型
        last_model_path = f'checkpoints/last_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        self.save_checkpoint(self.num_epochs, epoch_loss, epoch_accuracy, last_model_path, last_model=True)

        # 绘制训练损失和验证准确率图
        self.plot_training_results()


if __name__ == "__main__":
    # 加载数据
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实例化模型并移动到设备
    model = CNN().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建检查点目录（如果不存在）
    os.makedirs('checkpoints', exist_ok=True)

    # 设置日志文件路径
    log_file = 'logs/training.log'

    # 初始化Trainer对象并开始训练
    trainer = Trainer(model, criterion, optimizer, train_loader, test_loader, num_epochs=10, log_file=log_file, device=device)
    trainer.train()