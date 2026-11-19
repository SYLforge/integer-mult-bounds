"""Untimed, exhaustive differential check against the pinned legacy search."""

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CODEX = HERE.parents[1]
LEGACY = CODEX / 'multibank-latest-round-20261119'
INPUT = CODEX / 'gpu-cpu-search-pilot-20261119' / 'snapshot-172475c2' / 'complex'
KAPPA = '0.00387561167742'
PIN = '172475c2ca321d3496fda7a2a71478ada5acb948'
WORD_SHA = '341c49e9b9d6e8742f948cba2407f7ad67164f87f4191c21319730cfc8b419d3'
LAYOUT_SHA = '5bd55b71fd1757a2c907bdc2a32f69e29b38f960fa95b67780842502888e2364'
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy(directory):
    # Original scripts inject their directory into sys.path. The new search may
    # have its own bank_graph under the same name. Isolate both module objects.
    path_save = sys.path.copy()
    names = ('bank_graph', 'multibank_search')
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    try:
        spec = importlib.util.spec_from_file_location('pinned_fused_audit', directory / 'fused_search.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in names:
            require(Path(sys.modules[name].__file__).resolve().parent == directory,
                    f'{name} resolved outside the legacy source')
        return module
    finally:
        sys.path[:] = path_save
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def legacy_selection(fused, graph, layout, kappa, motif_limit, option_limit):
    groups, fixed, old, edges = fused.component_graph(graph, layout)
    neighbors = fused.incidence(old, edges)
    table = fused.price_tables(graph['frames'], kappa)[0][0]
    free = set(range(len(groups))) - fixed
    selected = []
    census = Counter()
    for cluster in fused.s.motifs(free, neighbors):
        info = fused.s.prepare(cluster, old, neighbors, table, option_limit)
        census[len(cluster)] += 1
        if info[-1] >= -info[4]:
            selected.append((info[-1], cluster, info))
    selected.sort(key=lambda row: (-row[0], row[1]))
    return selected[:motif_limit], len(selected), census, old, neighbors, table


def plan_record(row):
    upper, cluster, info = row
    options, exterior, internal, baseline, edgecount, info_upper = info
    return [int(upper), list(map(int, cluster)), [list(map(int, o)) for o in options],
            [list(map(int, e)) for e in exterior], [int(internal.get(p, 0)) for p in PAIRS],
            int(baseline), int(edgecount), int(info_upper)]


def check_plans(reference, new):
    require(len(reference) == len(new), 'selected motif count differs')
    digest = hashlib.sha256()
    for i, (left, right) in enumerate(zip(reference, new)):
        ref_record = plan_record(left)
        require(ref_record == plan_record(right), f'motif or prepared geometry differs at motif {i}')
        digest.update(json.dumps(ref_record, separators=(',', ':')).encode('ascii') + b'\n')
    return digest.hexdigest()


def legacy_fused_scores(fused, info, cluster, old, table, start, charts):
    options, exteriors, internal, baseline, _, _ = info
    size = len(charts)
    ordinal = np.arange(start, start + size, dtype=np.int64)
    base = np.full(size, -int(baseline), dtype=np.int64)
    for axis in reversed(range(len(options))):
        length = len(options[axis])
        base += exteriors[axis][ordinal % length]
        ordinal //= length
    padded = np.zeros((size, 4), dtype=np.int64)
    padded[:, :len(cluster)] = charts
    old_row = np.zeros(4, dtype=np.int64)
    old_row[:len(cluster)] = old[list(cluster)]
    weights = np.broadcast_to(np.array([internal.get(pair, 0) for pair in PAIRS], dtype=np.int64),
                              (size, len(PAIRS)))
    return fused.batch_score(padded, base, weights, np.broadcast_to(old_row, padded.shape), table, 'cpu')


def legacy_boundary(fused, graph, old, neighbors, kappa, motif, ordinal, score, cluster, chart, edges):
    # Independent scalar loop following distinct affected paid edges. In
    # addition, run fused_search.exact_patterns itself on this individual row.
    delta = [0] * 10
    new_charts = dict(zip(cluster, chart))
    dims = [len(f) for f in graph['frames']]
    for u in cluster:
        for kind, old_v, v, _ in neighbors[u]:
            require(kind == 0, 'unexpected helper edge')
            if v in new_charts and u > v:
                continue
            delta[abs(dims[int(old[u])] - dims[old_v])] += 1
            delta[abs(dims[new_charts[u]] - dims[new_charts.get(v, old_v)])] -= 1
    delta[0] = 0
    with localcontext() as ctx:
        ctx.prec = 90
        tariff = [Decimal(0)] + [Decimal(r) ** (1 - Decimal(kappa)) for r in range(1, 10)]
        credit = sum((n * h for n, h in zip(delta, tariff)), Decimal(0))
    seen = {}
    positives, n_unique = fused.exact_patterns(graph, cluster, old, neighbors,
                                                np.array([chart], dtype=np.int64), seen, kappa)
    positive = credit > Decimal('1e-70')
    require(n_unique == 1 and seen == {tuple(delta): credit} and bool(positives) == positive,
            f'legacy exact boundary disagrees with independent histogram at motif {motif}, row {ordinal}')
    return dict(motif_index=motif, local_ordinal=ordinal, cluster=list(cluster), charts=chart,
                score=int(score), edgecount=int(edges), delta=delta, credit=str(credit), positive=positive)


class OrderedRows:
    def __init__(self, fused, selected, graph, old, neighbors, table, kappa):
        self.fused = fused
        self.selected = selected
        self.graph = graph
        self.old = old
        self.neighbors = neighbors
        self.table = table
        self.kappa = kappa
        self.totals = [math.prod(map(len, info[0])) for _, _, info in selected]
        self.cursor = self.motif = self.offset = 0
        self.near = []
        self.charts_sha = hashlib.sha256()
        self.scores_sha = hashlib.sha256()

    def compare(self, start, charts, scores):
        require(int(start) == self.cursor, f'GPU audit chunks out of order at {self.cursor}')
        charts = np.asarray(charts)
        scores = np.asarray(scores)
        require(charts.shape == (len(scores), 4) and charts.dtype == np.int32,
                'GPU chart matrix has wrong shape or type')
        require(scores.ndim == 1 and scores.dtype == np.int64, 'GPU score vector has wrong shape or type')
        local = 0
        while local < len(scores):
            require(self.motif < len(self.selected), 'GPU produced extra rows')
            _, cluster, info = self.selected[self.motif]
            total = self.totals[self.motif]
            take = min(total - self.offset, len(scores) - local)
            reference = self.fused.s.assignments(info[0], self.offset, self.offset + take)
            # Price the actual graph edges via original scalar-bank CPU helper;
            # it is independent of both optimized and legacy fused weight packs.
            ref_scores = self.fused.s.scores_cpu(reference, cluster, self.old, self.neighbors, self.table)
            fused_scores = legacy_fused_scores(self.fused, info, cluster, self.old, self.table,
                                               self.offset, reference)
            require(np.array_equal(fused_scores, ref_scores),
                    f'legacy fused and independent incidence scorers differ at motif {self.motif}')
            section = slice(local, local + take)
            actual = charts[section]
            wrong_charts = np.any(actual[:, :len(cluster)] != reference, axis=1)
            if len(cluster) < 4:
                wrong_charts |= np.any(actual[:, len(cluster):] != 0, axis=1)
            mismatches = np.flatnonzero(wrong_charts | (scores[section] != ref_scores))
            if len(mismatches):
                j = int(mismatches[0])
                raise ValueError(f'row mismatch at motif {self.motif}, ordinal {self.offset+j}: '
                                 f'CPU {reference[j].tolist()} {ref_scores[j]}, '
                                 f'GPU {actual[j].tolist()} {scores[local+j]}')
            for j in np.flatnonzero(ref_scores >= -info[4] - 2):
                self.near.append(legacy_boundary(self.fused, self.graph, self.old, self.neighbors,
                                                 self.kappa, self.motif, self.offset + int(j),
                                                 ref_scores[j], cluster, reference[j].tolist(), info[4]))
            local += take
            self.offset += take
            if self.offset == total:
                self.motif += 1
                self.offset = 0
        self.charts_sha.update(np.ascontiguousarray(charts, dtype='<i4').tobytes())
        self.scores_sha.update(np.ascontiguousarray(scores, dtype='<i8').tobytes())
        self.cursor += len(scores)


def run(args):
    from prepare_fast import select_motifs
    from fast_search import boundary_record
    from gpu_score import score_selected

    legacy = args.legacy.resolve()
    fused = load_legacy(legacy)
    graph, layout = fused.read(args.graph), fused.read(args.layout)
    graph_sha, layout_sha = sha_file(args.graph), sha_file(args.layout)
    selection, survivors, census, old, neighbors, table = legacy_selection(
        fused, graph, layout, args.kappa, args.motifs, args.options)
    prep = select_motifs(graph, layout, args.kappa, args.motifs, args.options)
    require(census == prep.census and survivors == prep.nonexcluded, 'motif census differs')
    require(np.array_equal(old, prep.old) and np.array_equal(table, prep.table),
            'old charts or paid tariff table differ')
    plan_sha = check_plans(selection, prep.selected)
    check = OrderedRows(fused, selection, graph, old, neighbors, table, args.kappa)
    recovered = []

    def near(rows):
        for i, ordinal, score in rows:
            recovered.append(boundary_record(graph, prep, args.kappa, int(i), int(ordinal), int(score)))

    score_selected(prep.selected, prep.old, prep.table, near, on_audit=check.compare,
                   batch_size=args.batch)
    require(check.cursor == sum(check.totals) and check.motif == len(selection) and check.offset == 0,
            'GPU audit ended before visiting all ordinals')
    require(len(recovered) == len(check.near), 'GPU near list length differs')
    for ref, actual in zip(check.near, recovered):
        for key, expected in ref.items():
            value = actual[key]
            equal = Decimal(value) == Decimal(expected) if key == 'credit' else value == expected
            require(equal, f'boundary {ref["motif_index"]}/{ref["local_ordinal"]} {key} differs: '
                           f'expected {expected}, got {value}')
    fixed = (graph_sha == WORD_SHA and layout_sha == LAYOUT_SHA and args.kappa == KAPPA and
             args.options == 24 and args.motifs == 10000)
    if fixed:
        require(check.cursor == 27397025 and len(check.near) == 1 and not check.near[0]['positive'],
                'pinned reference result changed')
    return dict(status='PASS', benchmark_timing=False, pinned_input_match=fixed,
                commit=PIN if graph_sha == WORD_SHA and layout_sha == LAYOUT_SHA else None,
                graph_compressed_sha256=graph_sha, layout_compressed_sha256=layout_sha,
                canonical_graph_sha256=hashlib.sha256(fused.s.encode(graph)).hexdigest(),
                legacy_source_sha256={f: sha_file(legacy / f) for f in
                                      ['fused_search.py', 'multibank_search.py', 'bank_graph.py']},
                kappa=args.kappa, motif_census=dict(census), nonexcluded_motifs=survivors,
                selected_motifs=len(selection), options_limit=args.options, batch_size=args.batch,
                plan_sha256=plan_sha, rows_compared_elementwise=check.cursor,
                all_scores_matched_legacy_fused_cpu_and_graph_incidence_cpu=True,
                ordered_charts_sha256_le_int32_4_columns=check.charts_sha.hexdigest(),
                ordered_scores_sha256_le_int64=check.scores_sha.hexdigest(),
                boundary_records=check.near, positives=sum(row['positive'] for row in check.near),
                production_boundary_reconstruction_and_exact_verdict_matched=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, default=LEGACY)
    parser.add_argument('--graph', type=Path, default=INPUT / 'WORD.json.gz')
    parser.add_argument('--layout', type=Path, default=INPUT / 'LAYOUT.json.gz')
    parser.add_argument('--kappa', default=KAPPA)
    parser.add_argument('--motifs', type=int, default=10000)
    parser.add_argument('--options', type=int, default=24)
    parser.add_argument('--batch', type=int, default=524288)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(args)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
