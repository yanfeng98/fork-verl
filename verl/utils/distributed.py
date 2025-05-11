"""Utilities for distributed training."""
import os
import torch
from datetime import timedelta


def initialize_global_process_group(timeout_second: int = 36000) -> tuple[int, int, int]:
    """
    - local_rank: 当前 GPU 在本机上的编号
    - rank: 当前进程在整个集群中的编号
    - world_size: 总共有多少个 GPU/进程在参与训练
    """

    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    local_rank: int = int(os.environ["LOCAL_RANK"])
    rank: int = int(os.environ["RANK"])
    world_size: int = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size
