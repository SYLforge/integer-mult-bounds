"""Compact mixed-radix motif scoring with the CUDA Driver API and native NVRTC.

Only motif descriptors and the price table cross to the device in production.
Two independent CUDA streams feed exact CPU review while follow-on work is queued.
"""
import ctypes as C
from contextlib import ExitStack
import itertools
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


PAIRS = tuple(itertools.combinations(range(4), 2))
P = C.c_void_p
I = C.c_int
U = C.c_uint
D = C.c_uint64
Z = C.c_size_t
OUT = C.byref
SOURCE = Path(__file__).with_suffix(".cu")


def _bind(lib, name, *args):
    f = getattr(lib, name)
    f.argtypes = list(args)
    f.restype = I
    return f


def _checked(status, name):
    if status:
        raise RuntimeError(f"{name} failed with CUDA status {status}")


def _compile_native(arch):
    # PyTorch supplies these CUDA DLLs on the benchmark machine. Loading the
    # compiler directly avoids importing the PyTorch runtime into the new path.
    candidates = [Path(sys.executable).parent / "Lib/site-packages/torch/lib"]
    if os.environ.get("CUDA_PATH"):
        candidates.append(Path(os.environ["CUDA_PATH"]) / "bin")
    dll = next((p for d in candidates for p in sorted(d.glob("nvrtc64_*.dll"))
                if ".alt." not in p.name), None)
    if dll is None:
        raise RuntimeError(f"NVRTC shared library missing from {candidates}")
    with os.add_dll_directory(str(dll.parent)):
        # NVRTC loads its builtins by basename with the Windows legacy loader.
        # Preload the matching dependency from the runtime's own directory.
        builtin = next(dll.parent.glob("nvrtc-builtins64_*.dll"))
        builtins_library = C.WinDLL(str(builtin))
        lib = C.WinDLL(str(dll))
        create = _bind(lib, "nvrtcCreateProgram", C.POINTER(P), C.c_char_p,
                       C.c_char_p, I, P, P)
        compile_program = _bind(lib, "nvrtcCompileProgram", P, I,
                                C.POINTER(C.c_char_p))
        destroy = _bind(lib, "nvrtcDestroyProgram", C.POINTER(P))
        prog = P()
        _checked(create(OUT(prog), SOURCE.read_bytes(), b"gpu_score.cu", 0,
                        None, None), "nvrtcCreateProgram")
        try:
            flags = [f"--gpu-architecture={arch}".encode(), b"--std=c++14", b"-no-cache"]
            options = (C.c_char_p * len(flags))(*flags)
            status = compile_program(prog, len(options), options)
            if status:
                size = Z()
                _bind(lib, "nvrtcGetProgramLogSize", P, C.POINTER(Z))(prog, OUT(size))
                log = C.create_string_buffer(max(size.value, 1))
                _bind(lib, "nvrtcGetProgramLog", P, P)(prog, log)
                raise RuntimeError(f"NVRTC status {status}: {log.value.decode()}")
            size = Z()
            _checked(_bind(lib, "nvrtcGetCUBINSize", P, C.POINTER(Z))(prog, OUT(size)),
                     "nvrtcGetCUBINSize")
            binary = C.create_string_buffer(size.value)
            _checked(_bind(lib, "nvrtcGetCUBIN", P, P)(prog, binary), "nvrtcGetCUBIN")
            return binary
        finally:
            _checked(destroy(OUT(prog)), "nvrtcDestroyProgram")


class _Driver:
    def __init__(self, stack):
        lib = C.WinDLL("nvcuda.dll")
        signatures = {
            "cuInit": (U,), "cuDeviceGet": (C.POINTER(I), I),
            "cuDeviceGetAttribute": (C.POINTER(I), I, I),
            "cuDeviceGetName": (P, I, I),
            "cuDevicePrimaryCtxRetain": (C.POINTER(P), I),
            "cuDevicePrimaryCtxRelease_v2": (I,),
            "cuCtxPushCurrent_v2": (P,), "cuCtxPopCurrent_v2": (C.POINTER(P),),
            "cuModuleLoadData": (C.POINTER(P), P),
            "cuModuleGetFunction": (C.POINTER(P), P, C.c_char_p),
            "cuModuleUnload": (P,), "cuMemAlloc_v2": (C.POINTER(D), Z),
            "cuMemFree_v2": (D,), "cuMemAllocHost_v2": (C.POINTER(P), Z),
            "cuMemFreeHost": (P,), "cuMemcpyHtoD_v2": (D, P, Z),
            "cuMemcpyDtoHAsync_v2": (P, D, Z, P), "cuMemsetD32Async": (D, U, Z, P),
            "cuStreamCreate": (C.POINTER(P), U), "cuStreamDestroy_v2": (P,),
            "cuStreamSynchronize": (P,),
            "cuLaunchKernel": (P, U, U, U, U, U, U, U, P, C.POINTER(P), P),
        }
        self.functions = {name: _bind(lib, name, *params)
                          for name, params in signatures.items()}
        self.stack = stack
        device, context = I(), P()
        self.call("cuInit", 0)
        self.call("cuDeviceGet", OUT(device), 0)
        name = C.create_string_buffer(256)
        self.call("cuDeviceGetName", name, len(name), device)
        self.name = name.value.decode()
        self.call("cuDevicePrimaryCtxRetain", OUT(context), device)
        stack.callback(self.call, "cuDevicePrimaryCtxRelease_v2", device)
        self.call("cuCtxPushCurrent_v2", context)
        stack.callback(self.call, "cuCtxPopCurrent_v2", OUT(P()))
        major, minor = I(), I()
        self.call("cuDeviceGetAttribute", OUT(major), 75, device)
        self.call("cuDeviceGetAttribute", OUT(minor), 76, device)
        self.arch = f"sm_{major.value}{minor.value}"

    def call(self, name, *args):
        _checked(self.functions[name](*args), name)

    def alloc(self, size):
        ptr = D()
        self.call("cuMemAlloc_v2", OUT(ptr), max(int(size), 1))
        self.stack.callback(self.call, "cuMemFree_v2", ptr)
        return ptr

    def upload(self, array):
        ptr = self.alloc(array.nbytes)
        self.call("cuMemcpyHtoD_v2", ptr, array.ctypes.data, array.nbytes)
        return ptr

    def pinned(self, shape, dtype):
        dtype = np.dtype(dtype)
        size = math.prod(shape) * dtype.itemsize
        ptr = P()
        self.call("cuMemAllocHost_v2", OUT(ptr), max(size, 1))
        self.stack.callback(self.call, "cuMemFreeHost", ptr)
        buf = (C.c_ubyte * size).from_address(ptr.value)
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    def stream(self):
        stream = P()
        self.call("cuStreamCreate", OUT(stream), 1)  # CU_STREAM_NON_BLOCKING
        self.stack.callback(self.call, "cuStreamDestroy_v2", stream)
        return stream

    def kernel(self, binary):
        module, fn = P(), P()
        self.call("cuModuleLoadData", OUT(module), binary)
        self.stack.callback(self.call, "cuModuleUnload", module)
        self.call("cuModuleGetFunction", OUT(fn), module, b"score_ordinals")
        return fn


def _pack(selected, old, table):
    # Layout is shared with gpu_score.cu: radices, offsets, original charts,
    # six unordered pair weights, then affected edge count.
    meta = np.zeros((len(selected), 19), dtype=np.int32)
    meta[:, :4] = 1
    options = [np.zeros(1, dtype=np.int32)]
    exterior = [np.zeros(1, dtype=np.int64)]
    prefix = np.zeros(len(selected) + 1, dtype=np.uint64)
    baseline = np.empty(len(selected), dtype=np.int64)
    cursor = 1
    total = 0
    for m, (_, cluster, info) in enumerate(selected):
        opts, ext, internal, original, edge_count, _ = info
        if not 3 <= len(cluster) <= 4:
            raise ValueError(f"unsupported motif length {len(cluster)}")
        sizes = tuple(map(len, opts))
        for j, (ids, prices) in enumerate(zip(opts, ext, strict=True)):
            meta[m, j] = sizes[j]
            meta[m, 4+j] = cursor
            meta[m, 8+j] = old[cluster[j]]
            options.append(ids)
            exterior.append(prices)
            cursor += sizes[j]
        meta[m, 12:18] = [internal.get(pair, 0) for pair in PAIRS]
        meta[m, 18] = edge_count
        baseline[m] = original
        total += math.prod(sizes)
        prefix[m+1] = total
    if cursor > np.iinfo(np.int32).max or total > np.iinfo(np.int64).max:
        raise ValueError("packed descriptor exceeds CUDA indexing capacity")
    arrays = (prefix, meta, baseline, np.concatenate(options).astype(np.int32, copy=False),
              np.concatenate(exterior).astype(np.int64, copy=False),
              np.ascontiguousarray(table, dtype=np.int64))
    return arrays, total


class _Slot:
    def __init__(self, drv, capacity, audit):
        self.drv = drv
        self.stream = drv.stream()
        self.host_count = drv.pinned((1,), np.uint32)
        self.host_hits = drv.pinned((capacity, 3), np.int64)
        self.count = drv.alloc(self.host_count.nbytes)
        self.hits = drv.alloc(self.host_hits.nbytes)
        self.charts = D()
        self.scores = D()
        if audit:
            self.host_charts = drv.pinned((capacity, 4), np.int32)
            self.host_scores = drv.pinned((capacity,), np.int64)
            self.charts = drv.alloc(self.host_charts.nbytes)
            self.scores = drv.alloc(self.host_scores.nbytes)

    def copy_back(self, host, device):
        self.drv.call("cuMemcpyDtoHAsync_v2", host.ctypes.data, device,
                      host.nbytes, self.stream)

    def submit(self, fn, start, size, motifs, chart_count, descriptors, audit):
        self.start, self.size = start, size
        self.drv.call("cuMemsetD32Async", self.count, 0, 1, self.stream)
        vals = [D(start), I(size), I(motifs), I(chart_count), *descriptors,
                self.count, self.hits, I(int(audit)), self.charts, self.scores]
        args = (P * len(vals))(*(C.addressof(v) for v in vals))
        self.drv.call("cuLaunchKernel", fn, (size+255)//256, 1, 1, 256, 1, 1,
                      0, self.stream, args, None)
        self.copy_back(self.host_count, self.count)
        if audit:
            self.copy_back(self.host_charts[:size], self.charts)
            self.copy_back(self.host_scores[:size], self.scores)

    def consume(self, on_near, on_audit):
        self.drv.call("cuStreamSynchronize", self.stream)
        count = int(self.host_count[0])
        if count:
            if count > self.size:
                raise RuntimeError("device reported more boundary hits than candidates")
            rows = self.host_hits[:count]
            self.copy_back(rows, self.hits)
            self.drv.call("cuStreamSynchronize", self.stream)
            on_near(sorted(map(tuple, rows.tolist())))
        if on_audit is not None:
            on_audit(self.start, self.host_charts[:self.size], self.host_scores[:self.size])
        return count


def score_selected(selected, old, table, on_near, *, on_audit=None, batch_size=524288):
    """Run all selected motifs, retaining every legacy integer boundary case.

    on_near gets [(motif_index, local_ordinal, score), ...], sorted per batch.
    on_audit(global_start, charts, scores) optionally gets all rows in order,
    with charts padded on the right by zero to 4 entries (valid during callback).
    """
    if not 1 <= batch_size <= 2**29-1:
        raise ValueError("batch_size out of range")
    start = perf_counter()
    arrays, total = _pack(selected, old, table)
    packed = perf_counter()
    stats = {"assignments_scored": total, "descriptor_packing_s": packed-start,
             "uploaded_descriptor_bytes": sum(a.nbytes for a in arrays),
             "batch_size": batch_size, "batch_count": 0, "boundary_hits": 0,
             "compilation": "native NVRTC CUBIN; -no-cache on every invocation"}
    if not total:
        return stats
    with ExitStack() as stack:
        drv = _Driver(stack)
        stats["device"] = drv.name
        stats["arch"] = drv.arch
        initialized = perf_counter()
        fn = drv.kernel(_compile_native(drv.arch))
        compiled = perf_counter()
        descriptors = [drv.upload(a) for a in arrays]
        audit = on_audit is not None
        capacity = min(total, batch_size)
        slots = [_Slot(drv, capacity, audit) for _ in range(2)]
        ready = perf_counter()
        batches = range(0, total, batch_size)
        stats["batch_count"] = len(batches)

        # Launch the next batch before waiting on / CPU-pricing the previous
        # batch. Streams, counters, scratch space and pinned buffers are disjoint.
        for i, first in enumerate(batches):
            slots[i % 2].submit(fn, first, min(batch_size, total-first), len(selected),
                                table.shape[0], descriptors, audit)
            if i:
                stats["boundary_hits"] += slots[(i-1) % 2].consume(on_near, on_audit)
        stats["boundary_hits"] += slots[(len(batches)-1) % 2].consume(on_near, on_audit)
        finished = perf_counter()
        stats.update(driver_init_s=initialized-packed, native_compilation_s=compiled-initialized,
                     upload_and_buffers_s=ready-compiled, pipeline_wall_s=finished-ready)
    stats["scorer_wall_s"] = perf_counter()-start
    return stats
