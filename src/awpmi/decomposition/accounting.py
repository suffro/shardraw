"""Block-granular I/O accounting for row-major stored levels.

Logical byte counts charge exactly the rows read. A storage device reads whole
blocks, so this module also counts the distinct `block_bytes` blocks that a set of
rows of a row-major file touches. Used by the Phase 1B oracle and the Phase 1C
runtime alike.
"""

from __future__ import annotations

import torch

IO_BLOCK_BYTES = 4096


def blocks_touched(rows: torch.Tensor, row_bytes: int, block_bytes: int = IO_BLOCK_BYTES) -> torch.Tensor:
    """Distinct `block_bytes` blocks read per column when the selected rows [V, B] of a row-major file are read."""
    count = rows.shape[1]
    if row_bytes == 0:
        return torch.zeros(count, dtype=torch.int64, device=rows.device)
    if row_bytes > block_bytes:
        raise ValueError("rows larger than a block are not supported")
    index = torch.arange(rows.shape[0], device=rows.device)
    first = (index * row_bytes) // block_bytes
    last = ((index + 1) * row_bytes - 1) // block_bytes
    blocks = torch.arange(int(last[-1]) + 1, device=rows.device)
    prefix = torch.cat([torch.zeros(1, count, dtype=torch.int64, device=rows.device), rows.long().cumsum(dim=0)])

    def selected_with(key: torch.Tensor) -> torch.Tensor:
        # `key` is non-decreasing, so the rows whose key is b form one contiguous range.
        start = torch.searchsorted(key, blocks, right=False)
        stop = torch.searchsorted(key, blocks, right=True)
        return prefix[stop] - prefix[start]

    # A row no larger than a block touches only its first and last blocks.
    return ((selected_with(first) > 0) | (selected_with(last) > 0)).sum(dim=0)
