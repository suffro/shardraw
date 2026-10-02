"""Positioned reads of files (pread semantics), with or without the OS page cache.

`PositionedFile.read_into(offset, length, address)` reads `length` bytes at `offset` into
host memory at `address`. It is thread-safe: a pool of threads can keep several reads in
flight, which is how a solid-state drive reaches its throughput on small random reads.

direct=True bypasses the page cache (Windows `FILE_FLAG_NO_BUFFERING`, Linux `O_DIRECT`).
Every read then goes to the device, and the bytes it returns are the bytes the device
transferred. Offsets, lengths and buffer addresses must be multiples of
`DIRECT_ALIGNMENT`. With direct=False the OS may serve a read from its cache or read
ahead of it; the bytes requested are still exact, the bytes the device moved are not
known.

The module also reports what the OS says this process read (`os_read_counters`) and how
much memory it holds (`process_memory`), so a store's own accounting can be checked
against an independent source.

The Windows implementation (ctypes, kernel32) is the one this repository runs and tests.
The POSIX one (os.preadv, /proc/self) follows the same contract.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path

import torch

DIRECT_ALIGNMENT = 4096

if sys.platform == "win32":
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _GENERIC_READ = 0x80000000
    _SHARE_ALL = 0x1 | 0x2 | 0x4  # read, write, delete
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_NORMAL = 0x80
    _FILE_FLAG_NO_BUFFERING = 0x20000000
    _INVALID_HANDLE = ctypes.c_void_p(-1).value

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
    ]
    _kernel32.ReadFile.restype = wintypes.BOOL
    _kernel32.ReadFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_OVERLAPPED)
    ]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetProcessIoCounters.restype = wintypes.BOOL
    _kernel32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(_IO_COUNTERS)]
    _kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    _kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wintypes.DWORD
    ]

_MAX_READ = 1 << 30  # one ReadFile / preadv call reads at most this much


class PositionedFile:
    """Thread-safe positioned reads of one file; `direct` bypasses the OS page cache."""

    def __init__(self, path: str | Path, direct: bool) -> None:
        self.path = str(path)
        self.direct = direct
        self.size = os.path.getsize(self.path)
        self._lock = threading.Lock()
        self._closed = False
        if sys.platform == "win32":
            # A handle opened for synchronous I/O serializes its operations, so each thread gets its own.
            self._local = threading.local()
            self._handles: list[int] = []
        else:
            flags = os.O_RDONLY
            if direct:
                if not hasattr(os, "O_DIRECT"):
                    raise OSError("direct I/O is not supported on this platform")
                flags |= os.O_DIRECT
            self._fd = os.open(self.path, flags)

    def _handle(self) -> int:
        handle = getattr(self._local, "handle", None)
        if handle is None:
            flags = _FILE_ATTRIBUTE_NORMAL | (_FILE_FLAG_NO_BUFFERING if self.direct else 0)
            handle = _kernel32.CreateFileW(self.path, _GENERIC_READ, _SHARE_ALL, None, _OPEN_EXISTING, flags, None)
            if handle in (None, _INVALID_HANDLE):
                raise ctypes.WinError(ctypes.get_last_error())
            with self._lock:
                if self._closed:
                    _kernel32.CloseHandle(handle)
                    raise ValueError(f"{self.path} is closed")
                self._handles.append(handle)
            self._local.handle = handle
        return handle

    def read_into(self, offset: int, length: int, address: int) -> int:
        """Read `length` bytes at `offset` into `address`; returns the bytes read (fewer only at end of file)."""
        if self._closed:
            raise ValueError(f"{self.path} is closed")
        if self.direct and (offset % DIRECT_ALIGNMENT or length % DIRECT_ALIGNMENT or address % DIRECT_ALIGNMENT):
            raise ValueError("direct reads need aligned offsets, lengths and buffers")
        done = 0
        while done < length:
            chunk = min(length - done, _MAX_READ)
            count = self._read_once(offset + done, chunk, address + done)
            done += count
            if count < chunk:
                break  # end of file
        return done

    def _read_once(self, offset: int, length: int, address: int) -> int:
        if sys.platform == "win32":
            overlapped = _OVERLAPPED()
            overlapped.Offset = offset & 0xFFFFFFFF
            overlapped.OffsetHigh = offset >> 32
            count = wintypes.DWORD()
            ok = _kernel32.ReadFile(
                self._handle(), ctypes.c_void_p(address), length, ctypes.byref(count), ctypes.byref(overlapped)
            )
            if not ok:
                error = ctypes.get_last_error()
                if error == 38:  # ERROR_HANDLE_EOF
                    return 0
                raise ctypes.WinError(error)
            return count.value
        target = (ctypes.c_char * length).from_address(address)
        if hasattr(os, "preadv"):
            return os.preadv(self._fd, [target], offset)
        data = os.pread(self._fd, length, offset)
        ctypes.memmove(address, data, len(data))
        return len(data)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if sys.platform == "win32":
                for handle in self._handles:
                    _kernel32.CloseHandle(handle)
                self._handles.clear()
            else:
                os.close(self._fd)

    def __enter__(self) -> PositionedFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def os_read_counters() -> tuple[int, int] | None:
    """(read operations, bytes read) of this process as the OS counts them, or None if unavailable.

    Windows: `GetProcessIoCounters` (every ReadFile, cached or not). Linux: /proc/self/io
    (syscr, rchar). A direct read is one operation of exactly the requested bytes.
    """
    if sys.platform == "win32":
        counters = _IO_COUNTERS()
        if not _kernel32.GetProcessIoCounters(_kernel32.GetCurrentProcess(), ctypes.byref(counters)):
            return None
        return int(counters.ReadOperationCount), int(counters.ReadTransferCount)
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())
        return int(fields["syscr"]), int(fields["rchar"])
    except (OSError, KeyError, ValueError):
        return None


def process_memory() -> dict[str, int] | None:
    """Resident memory of this process: current and peak working set (Windows) or RSS and high-water mark (Linux)."""
    if sys.platform == "win32":
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not _kernel32.K32GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return {
            "resident_bytes": int(counters.WorkingSetSize),
            "peak_resident_bytes": int(counters.PeakWorkingSetSize),
            "private_bytes": int(counters.PagefileUsage),
        }
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        kib = {key: int(fields[key].split()[0]) * 1024 for key in ("VmRSS", "VmHWM")}
        return {"resident_bytes": kib["VmRSS"], "peak_resident_bytes": kib["VmHWM"]}
    except (OSError, KeyError, ValueError):
        return None


def aligned_host_buffer(nbytes: int, alignment: int = DIRECT_ALIGNMENT, pin: bool | None = None) -> torch.Tensor:
    """A uint8 host buffer of `nbytes` whose address is a multiple of `alignment` (pinned when CUDA is available)."""
    pin = torch.cuda.is_available() if pin is None else pin
    raw = torch.empty(nbytes + alignment, dtype=torch.uint8, pin_memory=pin)
    shift = (-raw.data_ptr()) % alignment
    return raw[shift : shift + nbytes]
