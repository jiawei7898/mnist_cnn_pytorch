import torch
from torchvision import datasets, transforms


def load_data(batch_size, num_workers=0):
    """
    加载和预处理MNIST数据集。

    参数:
    batch_size (int): 批处理大小。
    num_workers (int): 数据加载器使用的线程数，默认为0（表示使用主进程）。

    返回:
    tuple: 训练数据加载器和测试数据加载器。
    """
    # 数据预处理：转换为Tensor并进行归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载训练数据集
    train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)

    # 下载并加载测试数据集
    test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers  # 传递 num_workers 参数
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers  # 传递 num_workers 参数
    )

    return train_loader, test_loader


# 使用示例
if __name__ == "__main__":
    batch_size = 64  # 可以根据自己的需要设置批处理大小
    num_workers = 4  # 根据你的CPU核心数调整
    train_loader, test_loader = load_data(batch_size, num_workers)
    print(f"Training set loaded with {len(train_loader)} batches.")
    print(f"Test set loaded with {len(test_loader)} batches.")