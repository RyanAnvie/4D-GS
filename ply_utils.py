# -*- coding: utf-8 -*-
"""Small PLY reader/writer for OpenMVG/OpenMVS point clouds.

Supports the vertex-only ASCII and binary_little_endian PLY files used by the
continue_frame dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


PLY_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _parse_header(f):
    header = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PLY header ended unexpectedly")
        text = line.decode("ascii", errors="replace").strip()
        header.append(text)
        if text == "end_header":
            break

    fmt = None
    vertex_count = 0
    props = []
    in_vertex = False
    for line in header:
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element "):
            parts = line.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            if parts[1] == "list":
                raise ValueError("List vertex properties are not supported")
            props.append((parts[2], parts[1]))
    return fmt, vertex_count, props


def read_vertex_ply(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        fmt, count, props = _parse_header(f)
        dtype = np.dtype([(name, PLY_DTYPE[typ]) for name, typ in props])
        if fmt == "binary_little_endian":
            data = np.fromfile(f, dtype=dtype, count=count)
        elif fmt == "ascii":
            rows = []
            names = [p[0] for p in props]
            types = [p[1] for p in props]
            for _ in range(count):
                vals = f.readline().decode("ascii").split()
                rows.append(tuple(float(v) if "float" in t or t == "double" else int(v) for v, t in zip(vals, types)))
            data = np.array(rows, dtype=dtype)
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    out = {name: data[name] for name, _ in props}
    return out


def write_vertex_ply(path: str | Path, fields: dict, text: bool = False) -> None:
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    names = list(fields.keys())
    count = len(next(iter(fields.values()))) if fields else 0
    arrays = {k: np.asarray(v) for k, v in fields.items()}

    type_names = {}
    for name, arr in arrays.items():
        if arr.dtype.kind == "f":
            type_names[name] = "float"
            arrays[name] = arr.astype(np.float32)
        elif arr.dtype.kind == "u" and arr.dtype.itemsize == 1:
            type_names[name] = "uchar"
            arrays[name] = arr.astype(np.uint8)
        elif arr.dtype.kind in ("i", "u"):
            type_names[name] = "int"
            arrays[name] = arr.astype(np.int32)
        else:
            raise ValueError(f"Unsupported PLY field dtype for {name}: {arr.dtype}")

    with path.open("wb") as f:
        fmt = "ascii" if text else "binary_little_endian"
        lines = ["ply", f"format {fmt} 1.0", f"element vertex {count}"]
        lines += [f"property {type_names[name]} {name}" for name in names]
        lines.append("end_header")
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        if text:
            for i in range(count):
                f.write((" ".join(str(arrays[name][i].item()) for name in names) + "\n").encode("ascii"))
        else:
            dtype = np.dtype([(name, arrays[name].dtype) for name in names])
            data = np.empty(count, dtype=dtype)
            for name in names:
                data[name] = arrays[name]
            data.tofile(f)
