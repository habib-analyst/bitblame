"""Read a remote parquet file via HTTP Range requests (no full download).

Wraps a URL in a seekable, readable file-like object backed by curl range
fetches with an LRU block cache; pyarrow.parquet.ParquetFile then reads
only the footer + the row groups actually accessed.
"""
from __future__ import annotations

import io
import subprocess


class RangeReader(io.RawIOBase):
    BLOCK = 1 << 20  # 1 MB blocks

    def __init__(self, url: str, size: int):
        self.url = url
        self._size = size
        self._pos = 0
        self._cache: dict[int, bytes] = {}

    def _fetch(self, start: int, end: int) -> bytes:
        # fetch [start, end) via whole overlapping blocks
        b0, b1 = start // self.BLOCK, (end - 1) // self.BLOCK
        out = bytearray()
        for b in range(b0, b1 + 1):
            if b not in self._cache:
                bs, be = b * self.BLOCK, min((b + 1) * self.BLOCK, self._size) - 1
                r = subprocess.run(
                    ["curl", "-s", "--retry", "2", "--max-time", "120",
                     "-r", f"{bs}-{be}", self.url],
                    capture_output=True)
                r.check_returncode()
                self._cache[b] = r.stdout
                # bound the cache
                if len(self._cache) > 64:
                    self._cache.pop(next(iter(self._cache)))
            out += self._cache[b]
        base = b0 * self.BLOCK
        return bytes(out[start - base:end - base])

    def readable(self): return True
    def seekable(self): return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self._pos = offset
        elif whence == io.SEEK_CUR: self._pos += offset
        elif whence == io.SEEK_END: self._pos = self._size + offset
        return self._pos

    def tell(self): return self._pos

    def read(self, n=-1):
        if n is None or n < 0: n = self._size - self._pos
        n = min(n, self._size - self._pos)
        if n <= 0: return b""
        data = self._fetch(self._pos, self._pos + n)
        self._pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def resolve(url: str) -> tuple[str, int]:
    """Follow redirects to the signed CDN URL; return (url, size).

    Size comes from the final response's Content-Length header (HEAD
    requests report size_download=0, so the header is parsed instead).
    """
    r = subprocess.run(
        ["curl", "-sI", "--max-time", "60", "-L", url],
        capture_output=True, text=True)
    r.check_returncode()
    # last Effective URL: re-request with -w on GET of 0 bytes is simpler
    r2 = subprocess.run(
        ["curl", "-sL", "--max-time", "60", "-o", "/dev/null",
         "-w", "%{url_effective} %{size_download}", "-r", "0-0", url],
        capture_output=True, text=True)
    r2.check_returncode()
    final_url, first = r2.stdout.strip().rsplit(" ", 1)
    # size via a HEAD on the final URL
    r3 = subprocess.run(
        ["curl", "-sI", "--max-time", "60", final_url],
        capture_output=True, text=True)
    size = 0
    for line in r3.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            size = int(line.split(":", 1)[1].strip())
    if size <= 0:
        raise RuntimeError("could not determine remote file size")
    return final_url, size
