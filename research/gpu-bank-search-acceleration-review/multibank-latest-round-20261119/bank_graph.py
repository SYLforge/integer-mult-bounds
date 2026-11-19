# Adapted from our prior 5080 search/component_screen.py, preserved unchanged at source.
# New round: all public boundary components fixed; GPU proposals are not proofs.
"""Screen tied scalar-bank assignments on a P104 supplier graph and its public layout."""

import argparse
import gzip
import json
import time
from collections import defaultdict
from decimal import Decimal, localcontext
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

KAPPA = "0.00387290895338"
UNIT = 10**10
INVALID = -(10**14)
PAID, HELPER_START, HELPER_END = range(3)
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def span_mask(frame):
    values = {0}
    for axis in frame:
        values |= {x ^ axis for x in values}
    return sum(1 << x for x in values)


def component_graph(graph, layout):
    """Each node identifies scalar-form inputs/outputs that must use one chart."""
    links = {}
    charts = {}

    def find(node):
        while links[node] != node:
            links[node] = links[links[node]]
            node = links[node]
        return node

    def node(side, key, chart):
        pair = (side, key)
        links[pair] = pair
        charts[pair] = chart
        return pair

    edges = []
    fixed = set()
    mains = set(layout["live"]) | {r + 112 for r in layout["live"]}
    mains |= set(layout["hosted"]["X2"]) | set(layout["hosted"]["Y2"])
    for k, a, b in graph["moves"]:
        edges.append((PAID, node("form", k, a), node("value", k, b), k))
    for r, (a, b) in enumerate(zip(graph["initial"], graph["final"], strict=True)):
        start = node("value", ~r, a)
        stop = node("form", graph["first_output"] + r, b)
        if True:  # This round freezes EVERY public owner boundary.
            fixed.update([start, stop])
        else:
            edges.append((HELPER_START, start, stop, r))
    for k, terms in graph["forms"]:
        u = ("form", k)
        for j, _ in terms:
            v = ("value", j)
            assert charts[u] == charts[v]
            links[find(v)] = find(u)
    classes = defaultdict(list)
    for key in links:
        classes[find(key)].append(key)
    groups = sorted(sorted(group) for group in classes.values())
    indices = {key: ix for ix, group in enumerate(groups) for key in group}
    fixed_groups = {indices[key] for key in fixed}
    old = np.asarray([charts[group[0]] for group in groups])
    links = [(kind, indices[u], indices[v], k) for kind, u, v, k in edges]
    return groups, fixed_groups, old, links


def price_tables(frames, kappa=KAPPA):
    masks = [span_mask(frame) for frame in frames]
    dims = np.array(list(map(len, frames)))
    # Under the native tariff, nesting of actual spaces, not equal rank, is decisive.
    subset = np.array([[(a & b) == a and a != b for b in masks] for a in masks])
    rank = np.abs(dims[:, None] - dims)
    with localcontext() as ctx:
        ctx.prec = 60
        precise = [Decimal(0)] + [
            Decimal(r) ** (1 - Decimal(kappa)) for r in range(1, max(dims) + 1)
        ]
        units = np.array([int((v * UNIT).to_integral_value()) for v in precise])
    values = units[rank]
    paid = np.where(subset | subset.T, -values, INVALID)
    np.fill_diagonal(paid, 0)
    return np.stack(
        [paid, np.where(subset, values, INVALID), np.where(subset.T, values, INVALID)]
    ), precise


def incidence(old, edges):
    neighbors = [[] for _ in old]
    for kind, u, v, key in edges:
        assert u != v
        neighbors[u].append((kind, int(old[v]), v, key))
        neighbors[v].append((PAID if kind == PAID else HELPER_END, int(old[u]), u, key))
    return neighbors


def run_device(table, old, free, neighbors, device, limit):
    # Include upload of host inputs, construction of the cartesian chart candidates,
    # all feasibility tests and score reductions, ranking, and return from device.
    start = time.perf_counter()
    dt = torch.as_tensor(table, device=device)
    result = np.full((len(old), table.shape[-1]), INVALID, dtype=np.int64)
    bins = defaultdict(list)
    for u in free:
        bins[(len(neighbors[u]) - 1).bit_length()].append(u)
    batches = []
    for power, vertices in sorted(bins.items()):
        width = 1 << power
        m = table.shape[-1]
        per_batch = max(1, limit // (width * m * 48))
        for i in range(0, len(vertices), per_batch):
            batch = vertices[i : i + per_batch]
            kinds = np.zeros((len(batch), width), dtype=np.int64)
            adj = np.zeros_like(kinds)
            active = np.zeros_like(kinds, dtype=bool)
            for pos, u in enumerate(batch):
                for j, (kind, chart, _, _) in enumerate(neighbors[u]):
                    kinds[pos, j] = kind
                    adj[pos, j] = chart
                    active[pos, j] = True
            k = torch.as_tensor(kinds, device=device)
            a = torch.as_tensor(adj, device=device)
            active = torch.as_tensor(active, device=device)
            chart = torch.arange(m, device=device)
            # table[kind, proposed chart, existing adjacent chart]
            terms = dt[k[..., None], chart, a[..., None]]
            terms = torch.where(active[..., None], terms, 0)
            score = terms.sum(dim=1)
            valid = (terms != INVALID).all(dim=1)
            old_index = torch.as_tensor(old[batch], device=device)
            delta = score - score.gather(1, old_index[:, None])
            result[batch] = torch.where(valid, delta, INVALID).cpu().numpy()
            batches.append(
                {"vertices": len(batch), "charts": m, "padded_degree": width}
            )
    # Stable row-major tiebreak, shared by both devices. Includes host ranking.
    all_scores = result[free].ravel()
    order = np.argsort(-all_scores, kind="stable")
    ranked_digest = sha256(order.tobytes()).hexdigest()
    timing = time.perf_counter() - start
    return result, {
        "wall_seconds": timing,
        "candidates": len(all_scores),
        "per_second": len(all_scores) / timing,
        "ranking_sha256": ranked_digest,
        "batches": batches,
    }


def audit_boundary(samples, table, exact, frames, neighbors, old):
    controls = []
    with localcontext() as ctx:
        ctx.prec = 55
        for u, c, integer in samples:
            credit = Decimal(0)
            for kind, other, _, _ in neighbors[u]:
                initial = int(old[u])
                assert table[kind, c, other] != INVALID
                sign = -1 if kind == PAID else 1
                credit += sign * (
                    exact[abs(len(frames[c]) - len(frames[other]))]
                    - exact[abs(len(frames[initial]) - len(frames[other]))]
                )
            assert abs(credit * UNIT - integer) <= len(neighbors[u]) + Decimal("0.01")
            controls.append(
                {
                    "component": int(u),
                    "chart": int(c),
                    "gpu_integer": int(integer),
                    "decimal_credit": str(credit),
                }
            )
    return controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=HERE / "public-graph.json.gz")
    parser.add_argument("--layout", type=Path, default=HERE / "public-layout.json.gz")
    parser.add_argument("--output", type=Path, default=HERE / "component-screen.json")
    parser.add_argument("--kappa", default=KAPPA)
    args = parser.parse_args()
    common_start = time.perf_counter()
    g, layout = read(args.graph), read(args.layout)
    groups, fixed, old, edges = component_graph(g, layout)
    table, exact = price_tables(g["frames"], args.kappa)
    neighbors = incidence(old, edges)
    free = sorted(set(range(len(groups))) - fixed)
    common_seconds = time.perf_counter() - common_start
    avail, vram = torch.cuda.mem_get_info()
    limit = min(avail // 4, 650 * 1024**2)
    torch.set_num_threads(8)
    # Small matched run is also a real CUDA operation on graph-derived inputs.
    smoke_cpu, small_cpu = run_device(
        table, old, free[:8], neighbors, "cpu", 8 * 1024**2
    )
    smoke_gpu, small_gpu = run_device(
        table, old, free[:8], neighbors, "cuda", 8 * 1024**2
    )
    assert np.array_equal(smoke_cpu, smoke_gpu)
    del smoke_cpu, smoke_gpu
    cpu, timing_cpu = run_device(table, old, free, neighbors, "cpu", limit)
    gpu, timing_gpu = run_device(table, old, free, neighbors, "cuda", limit)
    assert (
        np.array_equal(cpu, gpu)
        and timing_cpu["ranking_sha256"] == timing_gpu["ranking_sha256"]
    )
    for result in (small_cpu, small_gpu, timing_cpu, timing_gpu):
        result["including_common_prepare_per_second"] = result["candidates"] / (
            common_seconds + result["wall_seconds"]
        )
    changes = np.argwhere(
        (cpu != INVALID) & (np.arange(table.shape[-1])[None, :] != old[:, None])
    )
    ranked = sorted(
        ((int(u), int(c), int(cpu[u, c])) for u, c in changes),
        key=lambda r: (-r[2], r[0], r[1]),
    )
    near = sorted(ranked, key=lambda row: (abs(row[2]), row[0], row[1]))[:8]
    positive = [v for v in ranked if v[2] > 0]
    negative = [v for v in ranked if v[2] < 0]
    sample = ranked[:5] + near + negative[:3] + positive[-3:]
    controls = audit_boundary(sample, table, exact, g["frames"], neighbors, old)
    report = {
        "graph_sha256": sha256(
            json.dumps(g, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "kappa": args.kappa,
        "unit": UNIT,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "vram_free": avail,
        "vram_total": vram,
        "batch_memory_budget": limit,
        "components": len(groups),
        "fixed_main_components": len(fixed),
        "free_components": len(free),
        "chart_count": len(g["frames"]),
        "common_preparation_seconds": common_seconds,
        "small_cpu": small_cpu,
        "small_gpu": small_gpu,
        "cpu": timing_cpu,
        "gpu": timing_gpu,
        "admissible_changes": len(ranked),
        "positive_changes": len(positive),
        "cpu_decimal_controls": controls,
        "top": [
            {
                "component": u,
                "chart": c,
                "before": int(old[u]),
                "credit_units": x,
                "members": groups[u],
            }
            for u, c, x in positive[:50]
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            k: report[k]
            for k in (
                "components",
                "free_components",
                "chart_count",
                "admissible_changes",
                "positive_changes",
            )
        }
    )
    print("Best changes:", ranked[:10])


if __name__ == "__main__":
    main()

