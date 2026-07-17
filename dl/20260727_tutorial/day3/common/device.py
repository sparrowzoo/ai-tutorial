import torch


def get_device():
    """
    返回当前可用的最佳计算设备

    Returns:
        torch.device: 当前设备对象
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


# 使用示例
device = get_device()
print("Using device:", device)