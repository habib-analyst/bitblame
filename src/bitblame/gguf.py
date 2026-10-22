"""GGUF header reader — pure-Python, no llama.cpp tool dependency.

Why this exists (P2 gap 2): the closed-loop harness verified the recipe
GGUF with ``llama-gguf <file> r``, which runs the example tool's hardcoded
self-test (``gguf_ex_read_1`` expects ``data[0] == 100.0``) and aborts with
``GGML_ASSERT`` on ANY real model file — the file was fine, the checker was
not. Verification now parses the GGUF header directly: magic, metadata KV,
and the per-tensor table (name, dims, ggml type, offset). No data section
is read, so this is fast and cannot trip over tensor contents.

GGUF layout (v3): magic "GGUF" | u32 version | u64 tensor_count |
u64 metadata_kv_count | metadata entries | tensor infos | data section.
Metadata values: 0 UINT8, 1 INT8, 2 UINT16, 3 INT16, 4 UINT32, 5 INT32,
6 FLOAT32, 7 BOOL, 8 STRING, 9 ARRAY, 10 UINT64, 11 INT64, 12 FLOAT64.
"""
from __future__ import annotations

import math
import struct

# ggml_type enum (ggml.h) -> (name, block_size, bytes_per_block).
# Only types relevant to this repo are listed; unknown codes raise.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    4: ("Q4_2", 32, 20),  # legacy; size per ggml block layout
    5: ("Q4_3", 32, 22),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("I8", 1, 1),
    17: ("I16", 1, 2),
    18: ("I32", 1, 4),
    19: ("I64", 1, 8),
    20: ("F64", 1, 8),
    21: ("I1", 1, 1),
    22: ("BF16", 1, 2),
}

NAME_TO_CODE = {v[0]: k for k, v in GGML_TYPES.items()}


def bytes_per_param(type_name: str) -> float:
    """Stored bytes per element for a ggml type (block bytes / block size)."""
    for _, (name, blk, nbytes) in GGML_TYPES.items():
        if name == type_name:
            return nbytes / blk
    raise ValueError(f"unknown ggml type {type_name!r}")


def block_size(type_name: str) -> int:
    for _, (name, blk, _) in GGML_TYPES.items():
        if name == type_name:
            return blk
    raise ValueError(f"unknown ggml type {type_name!r}")


class GGUFError(ValueError):
    pass


def _read_string(buf: bytes, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    s = buf[off:off + n].decode("utf-8")
    return s, off + n


def _read_value(buf: bytes, off: int, vtype: int):
    if vtype == 0:
        return buf[off], off + 1
    if vtype == 1:
        return struct.unpack_from("<b", buf, off)[0], off + 1
    if vtype == 2:
        return struct.unpack_from("<H", buf, off)[0], off + 2
    if vtype == 3:
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if vtype == 4:
        return struct.unpack_from("<I", buf, off)[0], off + 4
    if vtype == 5:
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if vtype == 6:
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if vtype == 7:
        return bool(buf[off]), off + 1
    if vtype == 8:
        return _read_string(buf, off)
    if vtype == 9:
        (elem_type,) = struct.unpack_from("<I", buf, off)
        off += 4
        (n,) = struct.unpack_from("<Q", buf, off)
        off += 8
        vals = []
        for _ in range(n):
            v, off = _read_value(buf, off, elem_type)
            vals.append(v)
        return vals, off
    if vtype == 10:
        return struct.unpack_from("<Q", buf, off)[0], off + 8
    if vtype == 11:
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if vtype == 12:
        return struct.unpack_from("<d", buf, off)[0], off + 8
    raise GGUFError(f"unknown metadata value type {vtype}")


def read_gguf_header(path: str) -> dict:
    """Parse a GGUF header. Returns {version, metadata, tensors, data_start}.

    tensors: {name: {dims, n_elts, type (name), type_code, offset, n_bytes}}
    where n_bytes is the stored (block-quantized) data size and offset is
    relative to the start of the data section. Does NOT read tensor data.
    """
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:4] != b"GGUF":
        raise GGUFError(f"{path}: bad magic {buf[:4]!r}, not a GGUF file")
    (version,) = struct.unpack_from("<I", buf, 4)
    (tensor_count, kv_count) = struct.unpack_from("<QQ", buf, 8)
    off = 24
    metadata: dict = {}
    for _ in range(kv_count):
        key, off = _read_string(buf, off)
        (vtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        val, off = _read_value(buf, off, vtype)
        metadata[key] = val
    tensors: dict = {}
    for _ in range(tensor_count):
        name, off = _read_string(buf, off)
        (n_dims,) = struct.unpack_from("<I", buf, off)
        off += 4
        dims = struct.unpack_from("<" + "Q" * n_dims, buf, off)
        off += 8 * n_dims
        (type_code,) = struct.unpack_from("<I", buf, off)
        off += 4
        (toffset,) = struct.unpack_from("<Q", buf, off)
        off += 8
        if type_code not in GGML_TYPES:
            raise GGUFError(f"{path}: tensor {name}: unknown type code "
                            f"{type_code}")
        tname, blk, nbytes = GGML_TYPES[type_code]
        n_elts = 1
        for d in dims:
            n_elts *= d
        n_blocks = math.ceil(n_elts / blk) if blk > 1 else n_elts
        tensors[name] = {
            "dims": list(dims),
            "n_elts": n_elts,
            "type": tname,
            "type_code": type_code,
            "offset": toffset,
            "n_bytes": n_blocks * nbytes,
        }
    alignment = int(metadata.get("general.alignment", 32))
    data_start = (off + alignment - 1) // alignment * alignment
    return {"version": version, "metadata": metadata, "tensors": tensors,
            "data_start": data_start, "header_bytes": off,
            "file_bytes": len(buf)}


def tensor_types(path: str) -> dict[str, str]:
    """{tensor name: ggml type name} for a GGUF file (header only)."""
    return {n: t["type"] for n, t in read_gguf_header(path)["tensors"].items()}
