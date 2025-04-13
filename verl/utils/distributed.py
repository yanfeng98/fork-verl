"""Utilities for distributed training."""
import os
import torch
from datetime import timedelta


def initialize_global_process_group(timeout_second=36000):

    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size
