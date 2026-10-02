"""PageCache: materialized pages kept on the device between requests, within a byte budget.

A cache entry is the device copy of one page (a segment row) or of a whole segment. The
budget is in bytes (DwarfStar's expert cache is likewise a memory budget for complete
pages, not a byte cache). Pinned entries are never evicted. On a miss the caller fetches
the page and offers it with `put`; the cache admits it if it fits after evicting
unpinned entries chosen by its replacement policy:

  LRUPolicy      least recently used first
  HotnessPolicy  lowest exponentially decayed access count first ("route hotness"): every
                 access adds 1, and a score halves every `half_life` accesses to the cache

`admit = False` freezes the contents: lookups still hit, misses are served and dropped.
DwarfStar does this during long prefills, where a request's working set is larger than
the cache and LRU would evict every page between its insertion and its next use.

The policy decides only *which* pages stay resident. It never changes a page's bytes,
so it affects efficiency only.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass

import torch


class ReplacementPolicy(ABC):
    @abstractmethod
    def touch(self, key: Hashable) -> None:
        """An access to a resident entry (or the insertion of a new one)."""

    @abstractmethod
    def forget(self, key: Hashable) -> None: ...

    @abstractmethod
    def victim(self, candidates: set) -> Hashable:
        """The entry to evict among `candidates` (resident, unpinned)."""

    def tick(self) -> None:
        """One access to the cache, hit or miss."""


class LRUPolicy(ReplacementPolicy):
    def __init__(self) -> None:
        self._order: OrderedDict = OrderedDict()

    def touch(self, key) -> None:
        self._order[key] = None
        self._order.move_to_end(key)

    def forget(self, key) -> None:
        self._order.pop(key, None)

    def victim(self, candidates: set):
        for key in self._order:
            if key in candidates:
                return key
        raise LookupError("no evictable entry")


class HotnessPolicy(ReplacementPolicy):
    """Exponentially decayed access counts; ties go to the least recently touched.

    Every touch gets a unique sequence number, so the victim never depends on the order in
    which candidates are iterated (a set of string keys iterates in a per-process hash order).
    """

    def __init__(self, half_life: float = 1024.0) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self.half_life = half_life
        self._now = 0
        self._sequence = 0
        self._score: dict = {}
        self._last: dict = {}
        self._touched: dict = {}

    def tick(self) -> None:
        self._now += 1

    def _decayed(self, key) -> float:
        return self._score.get(key, 0.0) * math.exp2(-(self._now - self._last.get(key, self._now)) / self.half_life)

    def touch(self, key) -> None:
        self._score[key] = self._decayed(key) + 1.0
        self._last[key] = self._now
        self._sequence += 1
        self._touched[key] = self._sequence

    def forget(self, key) -> None:
        # Keep the history: a page that comes back keeps its hotness.
        pass

    def victim(self, candidates: set):
        return min(candidates, key=lambda key: (self._decayed(key), self._touched.get(key, -1)))


POLICIES = {"lru": LRUPolicy, "hotness": HotnessPolicy}


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    hit_bytes: int = 0
    miss_bytes: int = 0
    inserts: int = 0
    insert_bytes: int = 0
    evictions: int = 0
    evicted_bytes: int = 0
    bypassed: int = 0
    bypassed_bytes: int = 0

    def reset(self) -> None:
        self.__init__()

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class PageCache:
    """Device-resident pages under a byte budget; see the module docstring."""

    def __init__(self, capacity_bytes: int, policy: ReplacementPolicy | None = None) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity_bytes = capacity_bytes
        self.policy = policy or LRUPolicy()
        self.admit = True
        self._entries: dict = {}
        self._pinned: set = set()
        self._namespaces: Counter = Counter()  # entries per key[0], for (namespace, item) keys
        self.pinned_bytes = 0
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.stats = CacheStats()

    def __contains__(self, key) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def peek(self, key) -> torch.Tensor | None:
        """The entry without counting an access."""
        return self._entries.get(key)

    def get(self, key, nbytes: int) -> torch.Tensor | None:
        """Look `key` up, counting a hit or a miss of `nbytes`."""
        self.stats.lookups += 1
        self.policy.tick()
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            self.stats.miss_bytes += nbytes
            return None
        self.stats.hits += 1
        self.stats.hit_bytes += nbytes
        self.policy.touch(key)
        return entry

    def get_many(self, keys: list, nbytes: int) -> list[torch.Tensor | None]:
        """`get` for several keys of `nbytes` each."""
        return [self.get(key, nbytes) for key in keys]

    def holds(self, namespace) -> bool:
        """Whether any entry has a key (namespace, ...)."""
        return self._namespaces[namespace] > 0

    def count_misses(self, pages: int, nbytes: int) -> None:
        """Count `pages` lookups known to miss (the caller checked `holds`), without looking each up."""
        self.stats.lookups += pages
        self.stats.misses += pages
        self.stats.miss_bytes += nbytes
        for _ in range(pages):
            self.policy.tick()

    def can_admit(self, nbytes: int) -> bool:
        """Whether a page of `nbytes` would be admitted now (possibly after evictions)."""
        return self.admit and nbytes <= self.capacity_bytes - self.pinned_bytes

    def bypass(self, pages: int, nbytes: int) -> None:
        """Count pages that were served without being offered (the caller knew they would not be admitted)."""
        self.stats.bypassed += pages
        self.stats.bypassed_bytes += nbytes

    def put(self, key, tensor: torch.Tensor) -> bool:
        """Offer a fetched page; returns whether it was admitted. The cache keeps `tensor` itself."""
        nbytes = tensor.numel() * tensor.element_size()
        if key in self._entries:
            return True
        if not self.can_admit(nbytes):
            self.bypass(1, nbytes)
            return False
        while self.resident_bytes + nbytes > self.capacity_bytes:
            self._evict(self.policy.victim(set(self._entries) - self._pinned))
        self._insert(key, tensor, nbytes)
        return True

    def pin(self, key, tensor: torch.Tensor) -> None:
        """Make `tensor` resident and never evict it (it counts against the budget)."""
        nbytes = tensor.numel() * tensor.element_size()
        if key in self._pinned:
            return
        if key not in self._entries:
            if nbytes > self.capacity_bytes - self.pinned_bytes:
                raise ValueError("pinned pages exceed the cache budget")
            while self.resident_bytes + nbytes > self.capacity_bytes:
                self._evict(self.policy.victim(set(self._entries) - self._pinned))
            self._insert(key, tensor, nbytes)
        self._pinned.add(key)
        self.pinned_bytes += nbytes

    def clear(self) -> None:
        self._entries.clear()
        self._pinned.clear()
        self._namespaces.clear()
        self.pinned_bytes = 0
        self.resident_bytes = 0

    def _insert(self, key, tensor: torch.Tensor, nbytes: int) -> None:
        self._entries[key] = tensor
        if isinstance(key, tuple) and key:
            self._namespaces[key[0]] += 1
        self.resident_bytes += nbytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        self.stats.inserts += 1
        self.stats.insert_bytes += nbytes
        self.policy.touch(key)

    def _evict(self, key) -> None:
        tensor = self._entries.pop(key)
        if isinstance(key, tuple) and key:
            self._namespaces[key[0]] -= 1
        nbytes = tensor.numel() * tensor.element_size()
        self.resident_bytes -= nbytes
        self.stats.evictions += 1
        self.stats.evicted_bytes += nbytes
        self.policy.forget(key)
