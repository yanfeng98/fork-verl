"""
Contains a resharding manager that binds weights from FSDP zero3 to XPerfGPT
"""
from .base import BaseShardingManager

from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.ulysses import get_ulysses_sequence_parallel_group, set_ulysses_sequence_parallel_group


class FSDPUlyssesShardingManager(BaseShardingManager):
    """
    Sharding manager to support data resharding when using FSDP + Ulysses
    """

    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()
        self.device_mesh: DeviceMesh = device_mesh
        self.seed_offset: int = 12345

    def __enter__(self):
        if self.device_mesh is not None:
            # We have a global SP group
            # so we have to change to use model-specific sp group
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.device_mesh['sp'].get_group())

    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        if self.device_mesh is not None:
            # revert to previous sp group
            set_ulysses_sequence_parallel_group(self.prev_sp_group)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        AllGather data from sp region
        This is because the data is first sharded along the FSDP dimension as we utilize the DP_COMPUTE
        In Ulysses, we need to make sure the same data is used across a SP group
        """
        if self.device_mesh is not None:
            group = self.device_mesh['sp'].get_group()
            all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """
        Split the data to follow FSDP partition
        """
        if self.device_mesh is not None:
            sp_size = self.device_mesh['sp'].size()
            sp_rank = self.device_mesh['sp'].get_local_rank()
            data = data.chunk(chunks=sp_size)[sp_rank]
        return data