"""Full RTX 5080 + CPU search. Only shortlist rows are reconstructed on the host."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bank_graph import read
from prepare_fast import select_motifs
from gpu_score import score_selected


def canonical_hash(obj):
    content = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def boundary_record(g, prep, kappa, motif_index, local_ordinal, score):
    _, cluster, info = prep.selected[motif_index]
    options, _, _, _, edgecount, _ = info
    ordinal = int(local_ordinal)
    charts = [0] * len(cluster)
    for axis in reversed(range(len(cluster))):
        ordinal, digit = divmod(ordinal, len(options[axis]))
        charts[axis] = int(options[axis][digit])
    if ordinal:
        raise ValueError("Boundary ordinal outside motif")

    proposed = dict(zip(cluster, charts))
    dimensions = [len(f) for f in g["frames"]]
    delta = [0] * 10
    for u in cluster:
        for _, old_chart, v, _ in prep.neighbors[u]:
            if v in proposed and v < u:
                continue
            before = abs(dimensions[prep.old[u]] - dimensions[old_chart])
            after = abs(dimensions[proposed[u]] - dimensions[proposed.get(v, old_chart)])
            delta[before] += 1
            delta[after] -= 1
    delta[0] = 0
    with localcontext() as ctx:
        ctx.prec = 90
        exponent = 1 - Decimal(kappa)
        credit = sum((n * Decimal(r) ** exponent for r, n in enumerate(delta) if r and n),
                     Decimal(0))
    return dict(motif_index=int(motif_index), local_ordinal=int(local_ordinal),
                cluster=list(map(int, cluster)), charts=charts, score=int(score),
                edgecount=int(edgecount), delta=delta, credit=str(credit),
                positive=credit > Decimal("1e-70"))


def run_round(g, layout, kappa, *, motif_limit=10000, option_limit=24,
              batch_size=524288, offset=0):
    start = time.perf_counter()
    prep = select_motifs(g, layout, kappa, motif_limit + offset, option_limit)
    prep.selected = prep.selected[offset:]
    prep_time = time.perf_counter() - start
    boundary = []
    exact_time = 0.0

    def exact(rows):
        nonlocal exact_time
        tick = time.perf_counter()
        for motif, ordinal, score in rows:
            boundary.append(boundary_record(g, prep, kappa, motif, ordinal, score))
        exact_time += time.perf_counter() - tick

    accounting = score_selected(prep.selected, prep.old, prep.table, exact,
                                batch_size=batch_size)
    positives = [b for b in boundary if b["positive"]]
    positives.sort(key=lambda b: -Decimal(b["credit"]))
    return {
        "baseline_graph": canonical_hash(g),
        "kappa": kappa,
        "total_connected_motifs": dict(prep.census),
        "nonexcluded_motifs": prep.nonexcluded,
        "selected_motifs": len(prep.selected),
        "selected_offset": offset,
        "options_limit": option_limit,
        "assignments_scored": accounting["assignments_scored"],
        "uncertain_or_positive_assignments_checked_by_exact_histogram": len(boundary),
        "unique_histograms": len({tuple(x["delta"]) for x in boundary}),
        "boundary_records": boundary,
        "positive_examples": positives,
        "common_preparation_s": prep_time,
        "preparation": prep.stats,
        "gpu": accounting,
        "exact_shortlisting_s": exact_time,
        "round_elapsed_s": time.perf_counter() - start,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--layout", required=True, type=Path)
    p.add_argument("--kappa", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--motifs", default=10000, type=int)
    p.add_argument("--options", default=24, type=int)
    p.add_argument("--offset", default=0, type=int)
    p.add_argument("--batch", type=int, default=524288)
    args = p.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    start = time.perf_counter()
    result = run_round(read(args.graph), read(args.layout), args.kappa,
                       motif_limit=args.motifs, option_limit=args.options,
                       batch_size=args.batch, offset=args.offset)
    result["observed_total_s"] = time.perf_counter() - start
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['assignments_scored']} assignments; {len(result['boundary_records'])} boundary; "
          f"{len(result['positive_examples'])} positive", flush=True)


if __name__ == "__main__":
    main()
