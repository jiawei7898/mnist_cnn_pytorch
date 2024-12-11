import torch


def check_cuda_availability():
    # 检查CUDA是否可用
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_cuda_available}")

    if is_cuda_available:
        # 获取当前默认的CUDA设备
        current_device = torch.cuda.current_device()
        print(f"Current CUDA Device: {current_device} (device id)")

        # 获取所有可用的CUDA设备的数量
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA Devices: {device_count}")

        # 获取每个CUDA设备的信息
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")
            capability = torch.cuda.get_device_capability(i)
            print(f"  - Capability: {capability[0]}.{capability[1]}")
            memory_info = torch.cuda.mem_get_info(i)
            total_memory = memory_info[1] / (1024 ** 3)  # Convert to GB
            free_memory = memory_info[0] / (1024 ** 3)  # Convert to GB
            print(f"  - Total Memory: {total_memory:.2f} GB")
            print(f"  - Free Memory: {free_memory:.2f} GB")

    else:
        print("No CUDA devices available. Training will be done on CPU.")


# 调用函数以检查CUDA
check_cuda_availability()
