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
# Scoring is implemented in gpu_score.py.

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
