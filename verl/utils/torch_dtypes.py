"""
Adapted from Cruise.
"""

import torch

HALF_LIST: list[int|str|torch.dtype] = [16, "16", "fp16", "float16", torch.float16]
FLOAT_LIST: list[int|str|torch.dtype] = [32, "32", "fp32", "float32", torch.float32]
BFLOAT_LIST: list[str|torch.dtype] = ["bf16", "bfloat16", torch.bfloat16]


class PrecisionType:
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    HALF: str = "16"
    FLOAT: str = "32"
    FULL: str = "64"
    BFLOAT: str = "bf16"
    MIXED: str = "mixed"

    @staticmethod
    def to_dtype(precision) -> torch.dtype:
        if precision in HALF_LIST:
            return torch.float16
        elif precision in FLOAT_LIST:
            return torch.float32
        elif precision in BFLOAT_LIST:
            return torch.bfloat16
        else:
            raise RuntimeError(f"unexpected precision: {precision}")

    @staticmethod
    def supported_type(precision: str | int) -> bool:
        return any(x == precision for x in PrecisionType)

    @staticmethod
    def supported_types() -> list[str]:
        return [x.value for x in PrecisionType]

    @staticmethod
    def is_fp16(precision):
        return precision in HALF_LIST

    @staticmethod
    def is_fp32(precision):
        return precision in FLOAT_LIST

    @staticmethod
    def is_bf16(precision):
        return precision in BFLOAT_LIST

    @staticmethod
    def to_str(precision):
        if precision == torch.float16:
            return "fp16"
        elif precision == torch.float32:
            return "fp32"
        elif precision == torch.bfloat16:
            return "bf16"
        else:
            raise RuntimeError(f"unexpected precision: {precision}")
