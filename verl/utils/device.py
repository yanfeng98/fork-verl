import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
        return False
    except ImportError:
        return False

is_cuda_available: bool = torch.cuda.is_available()
is_npu_available: bool = is_torch_npu_available()

def get_device_name() -> str:
    if is_cuda_available:
        device: str = "cuda"
    elif is_npu_available:
        device: str = "npu"
    else:
        device: str = "cpu"
    return device

def get_nccl_backend() -> str:
    if is_cuda_available:
        return "nccl"
    elif is_npu_available:
        return "hccl"
    else:
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")

def get_torch_device() -> Any:

    device_name: str = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda

def get_visible_devices_keyword() -> str:
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES' or `ASCEND_RT_VISIBLE_DEVICES`
    """
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_id() -> int:
    """Return current device id based on the device type.
    Returns:
        device index
    """
    return get_torch_device().current_device()

def set_expandable_segments(enable: bool) -> None:
    """Enable or disable expandable segments for cuda.
    Args:
        enable (bool): Whether to enable expandable segments. Used to avoid OOM.
    """
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
