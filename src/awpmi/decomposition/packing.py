"""Bit-packed storage of the integer codes of a refinement level.

A level with b-bit codes q ∈ [−limit, limit] (limit = 2^(b−1) − 1) stores each row
as one little-endian bitstream of ceil(K·b/8) bytes, which is the payload size the
byte accounting charges (`RefinementLevel.payload_bytes_per_row`). Code k holds the
biased value q + limit ∈ [0, 2·limit] in bits [k·b, (k+1)·b) of the row; bit i of a
row is bit (i mod 8) of byte ⌊i/8⌋. Rows are independent, so any subset of rows
can be read and decoded on its own.

Decoding is pure PyTorch: with b ≤ 8, a code spans at most two bytes, so one
16-bit window per code holds it entirely.
"""

from __future__ import annotations

import math

import torch

from awpmi.decomposition.quantization import code_limit

PACKED_DTYPE = torch.uint8
CODE_DTYPE = torch.int16
_PACK_CHUNK_ROWS = 4096


def payload_bytes(in_features: int, bits: int) -> int:
    return math.ceil(in_features * bits / 8)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer codes [V, K] (any integer dtype, |q| ≤ limit) into uint8 [V, ceil(K·b/8)]."""
    limit = code_limit(bits)
    if codes.dim() != 2 or codes.is_floating_point():
        raise TypeError("pack_codes expects an integer [rows, in_features] tensor")
    if codes.numel() and int(codes.abs().max()) > limit:
        raise ValueError(f"codes exceed ±{limit}, the range of {bits}-bit codes")
    rows, in_features = codes.shape
    width = payload_bytes(in_features, bits)
    packed = torch.empty(rows, width, dtype=PACKED_DTYPE, device=codes.device)
    bit_values = torch.arange(bits, device=codes.device)
    byte_weights = 2 ** torch.arange(8, device=codes.device)
    for start in range(0, rows, _PACK_CHUNK_ROWS):
        biased = codes[start : start + _PACK_CHUNK_ROWS].to(torch.int32) + limit
        # Bit t of code k sits at stream position k·b + t.
        stream = ((biased[..., None] >> bit_values) & 1).to(torch.uint8).reshape(biased.shape[0], -1)
        stream = torch.nn.functional.pad(stream, (0, width * 8 - stream.shape[1]))
        packed[start : start + _PACK_CHUNK_ROWS] = (stream.view(-1, width, 8) * byte_weights).sum(dim=-1).to(PACKED_DTYPE)
    return packed


class CodeLayout:
    """Byte positions and shifts of every code of a b-bit level with K codes per row."""

    def __init__(self, bits: int, in_features: int, device: torch.device | str) -> None:
        self.bits = bits
        self.limit = code_limit(bits)
        self.in_features = in_features
        self.width = payload_bytes(in_features, bits)
        position = torch.arange(in_features, device=device) * bits
        self.low_byte = position // 8
        # The high byte is only needed when a code crosses a byte boundary, in which
        # case it exists; otherwise clamping keeps the index in range harmlessly.
        self.high_byte = torch.clamp(self.low_byte + 1, max=self.width - 1)
        self.shift = (position % 8).to(CODE_DTYPE)
        self.mask = (1 << bits) - 1

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        """Signed codes [R, K] (int16) from packed rows [R, width] (uint8)."""
        if packed.dtype != PACKED_DTYPE or packed.dim() != 2 or packed.shape[1] != self.width:
            raise ValueError(f"expected uint8 [rows, {self.width}] packed rows")
        low = packed.index_select(1, self.low_byte).to(CODE_DTYPE)
        # Only bits shift..shift+b−1 ≤ 14 of the window are used, so dropping the high
        # byte's top bit keeps the int16 window non-negative without losing anything.
        high = (packed.index_select(1, self.high_byte) & 0x7F).to(CODE_DTYPE)
        window = low | (high << 8)
        return ((window >> self.shift) & self.mask) - self.limit


def unpack_codes(packed: torch.Tensor, bits: int, in_features: int) -> torch.Tensor:
    """Inverse of `pack_codes`: signed int16 codes [R, K]."""
    return CodeLayout(bits, in_features, packed.device).unpack(packed)
