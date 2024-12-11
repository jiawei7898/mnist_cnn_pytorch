# config.py

import os
from datetime import datetime
import torch

# 超参数
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 4  # 根据你的CPU核心数调整

# 路径
CHECKPOINT_DIR = '../checkpoints'
LOG_FILE = '../training.log'

# 检查点目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 当前时间戳
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')