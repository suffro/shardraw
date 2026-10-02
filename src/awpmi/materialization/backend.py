"""MaterializationBackend: page cache → page store → streamer, behind one call.

`materialize(segment, rows)` returns the requested rows (ascending) on the compute device as
raw bytes [n, row_bytes]:

  * a store that keeps its segments in memory on the compute device answers directly (the
    resident runtimes of Phases 1C and 2);
  * otherwise cached pages are taken from the `PageCache`, and the rest are fetched by the
    `PageStreamer` from the store and offered to the cache.

A whole-segment request is one cache entry; a request for some rows looks rows up one by one,
and also serves them from a cached whole segment. Every request is counted: rows and bytes
requested, served from the cache, fetched from storage; the store's and the streamer's own
counters give the physical bytes, reads and copies behind them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from awpmi.storage.cache import PageCache
from awpmi.storage.layout import Segment
from awpmi.storage.store import PageStore
from awpmi.streaming.streamer import PageStreamer, Ticket, resolve_device

WHOLE_SEGMENT = -1


@dataclass
class MaterializationStats:
    requests: int = 0
    rows: int = 0
    requested_bytes: int = 0
    cache_hit_rows: int = 0
    cache_hit_bytes: int = 0
    fetched_rows: int = 0  # served by the store (not the cache)
    fetched_bytes: int = 0
    device_copy_bytes: int = 0  # bytes copied on the device to assemble a request from cached pages

    def reset(self) -> None:
        self.__init__()

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class MaterializationBackend:
    def __init__(
        self,
        store: PageStore,
        device: torch.device | str,
        streamer: PageStreamer | None = None,
        cache: PageCache | None = None,
    ) -> None:
        self.store = store
        self.device = resolve_device(device)
        self.resident = store.in_memory and resolve_device(store.device) == self.device
        if not self.resident and streamer is None:
            raise ValueError("a store away from the compute device needs a streamer")
        if streamer is not None and streamer.device != self.device:
            raise ValueError("the streamer must deliver to the compute device")
        self.streamer = streamer
        self.cache = None if self.resident else cache
        self.stats = MaterializationStats()

    def segment(self, name: str) -> Segment:
        return self.store.segment(name)

    # Requests

    def materialize(self, segment: str, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Rows of `segment` on the device ([n, row_bytes] uint8, ascending rows). Read-only."""
        info = self.store.segment(segment)
        count = info.rows if rows is None else rows.numel()
        self.stats.requests += 1
        self.stats.rows += count
        self.stats.requested_bytes += count * info.row_bytes
        if self.resident:
            self.stats.fetched_rows += count
            self.stats.fetched_bytes += count * info.row_bytes
            return self.store.read_rows(segment, rows)
        if self.cache is None:
            return self._fetch(info, rows)
        whole = self.cache.peek((segment, WHOLE_SEGMENT))
        if rows is None:
            entry = self.cache.get((segment, WHOLE_SEGMENT), info.nbytes)
            if entry is not None:
                self._hit(info, info.rows)
                return entry
            data = self._fetch(info, None)
            self.cache.put((segment, WHOLE_SEGMENT), data)
            return data
        if whole is not None:
            self.cache.get((segment, WHOLE_SEGMENT), count * info.row_bytes)
            self._hit(info, count)
            data = whole.index_select(0, rows.to(self.device))
            self.stats.device_copy_bytes += count * info.row_bytes
            return data
        return self._rows_through_cache(info, rows)

    def _rows_through_cache(self, info: Segment, rows: torch.Tensor) -> torch.Tensor:
        if not self.cache.holds(info.name) and not self.cache.can_admit(info.row_bytes):
            # No row of this segment is cached, and none could be: every row misses and is not kept.
            count = rows.numel()
            self.cache.count_misses(count, count * info.row_bytes)
            self.cache.bypass(count, count * info.row_bytes)
            return self._fetch(info, rows)
        row_list = rows.reshape(-1).tolist()
        cached = self.cache.get_many([(info.name, row) for row in row_list], info.row_bytes)
        missing = [k for k, entry in enumerate(cached) if entry is None]
        hits = len(row_list) - len(missing)
        self._hit(info, hits)
        fetched = None
        if missing:
            fetched = self._fetch(info, torch.tensor([row_list[k] for k in missing], dtype=torch.int64))
            if self.cache.can_admit(info.row_bytes):
                for k, position in enumerate(missing):
                    self.cache.put((info.name, row_list[position]), fetched[k].clone())
            else:
                self.cache.bypass(len(missing), len(missing) * info.row_bytes)
            if not hits:
                return fetched
        out = torch.empty(len(row_list), info.row_bytes, dtype=torch.uint8, device=self.device)
        if missing:
            out.index_copy_(0, torch.tensor(missing, device=self.device), fetched)
        present = [k for k, entry in enumerate(cached) if entry is not None]
        out.index_copy_(0, torch.tensor(present, device=self.device), torch.stack([cached[k] for k in present]))
        self.stats.device_copy_bytes += hits * info.row_bytes
        return out

    def _hit(self, info: Segment, rows: int) -> None:
        self.stats.cache_hit_rows += rows
        self.stats.cache_hit_bytes += rows * info.row_bytes

    def _fetch(self, info: Segment, rows: torch.Tensor | None) -> torch.Tensor:
        count = info.rows if rows is None else rows.numel()
        self.stats.fetched_rows += count
        self.stats.fetched_bytes += count * info.row_bytes
        return self.streamer.fetch(self.store, info.name, rows)

    def pin(self, segment: str) -> None:
        """Keep a whole segment resident in the cache (fetched now, never evicted)."""
        if self.cache is None:
            raise ValueError("pinning needs a page cache")
        info = self.store.segment(segment)
        key = (segment, WHOLE_SEGMENT)
        if key not in self.cache:
            data = self._fetch(info, None)
            self.cache.pin(key, data)
        else:
            self.cache.pin(key, self.cache.peek(key))

    def prefetch(self, segment: str, rows: torch.Tensor | None = None) -> Ticket:
        """Start fetching rows in the background (not counted as requested until used)."""
        if self.resident:
            raise ValueError("a resident store needs no prefetch")
        return self.streamer.prefetch(self.store, segment, rows)

    # Accounting

    def reset_stats(self, record_ranges: bool = False) -> None:
        self.stats.reset()
        self.store.stats.reset(record_ranges)
        if self.streamer is not None:
            self.streamer.stats.reset()
        if self.cache is not None:
            self.cache.stats.reset()

    def report(self) -> dict:
        """Everything counted since the last reset (device copy time is waited for)."""
        report = {"materialization": self.stats.as_dict(), "storage": self.store.stats.as_dict()}
        report["storage"]["io_ms"] = self.store.stats.io_ms
        if self.streamer is not None:
            report["transfer"] = self.streamer.stats.as_dict()
            report["transfer"]["copy_ms"] = self.streamer.stats.copy_ms()
        if self.cache is not None:
            report["cache"] = self.cache.stats.as_dict()
            report["cache"]["resident_bytes"] = self.cache.resident_bytes
        return report

    @property
    def device_resident_bytes(self) -> int:
        """Bytes kept on the compute device between requests (a resident store, or the cache)."""
        if self.resident:
            return self.store.resident_bytes
        return 0 if self.cache is None else self.cache.resident_bytes

    @property
    def host_resident_bytes(self) -> int:
        """Bytes kept in host memory between requests (an in-memory host store, the streamer's buffers)."""
        nbytes = 0 if self.resident else self.store.resident_bytes
        return nbytes + (0 if self.streamer is None else self.streamer.host_resident_bytes)
