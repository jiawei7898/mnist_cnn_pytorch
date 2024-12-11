import torch


def check_gpu_for_training():
    # 检查CUDA是否可用
    is_cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {'Yes' if is_cuda_available else 'No'}")

    if not is_cuda_available:
        print("CUDA is not available. Cannot perform training on GPU.")
        return

    # 尝试创建一个张量并在GPU上运行一个简单的操作
    try:
        # 创建一个随机张量
        tensor = torch.randn(1000, 1000)

        # 将张量移动到GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = tensor.to(device)

        # 执行一个简单的矩阵乘法操作
        result = torch.matmul(tensor, tensor.T)

        # 将结果移回CPU并打印出来
        result_cpu = result.cpu()
        print("Matrix multiplication on GPU was successful.")
        print("Result of the first element:", result_cpu[0, 0].item())

        # 清理缓存
        del tensor, result, result_cpu
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred while trying to use the GPU for training: {e}")


# 调用函数以检查GPU是否可用于训练
check_gpu_for_training()
