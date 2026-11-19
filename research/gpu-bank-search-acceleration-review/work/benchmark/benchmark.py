"""Counterbalanced fresh-process full-path benchmark on the immutable 172475c2 input."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODEX = ROOT.parents[1]
LEGACY = CODEX / "multibank-latest-round-20261119"
SNAPSHOT = CODEX / "gpu-cpu-search-pilot-20261119/snapshot-172475c2"
FIXED = {
    "WORD.json.gz": "341c49e9b9d6e8742f948cba2407f7ad67164f87f4191c21319730cfc8b419d3",
    "LAYOUT.json.gz": "5bd55b71fd1757a2c907bdc2a32f69e29b38f960fa95b67780842502888e2364",
}
ORDER = [
    ("cpu", "cuda", "new"), ("cuda", "new", "cpu"), ("new", "cpu", "cuda"),
    ("new", "cuda", "cpu"), ("cpu", "new", "cuda"), ("cuda", "cpu", "new"),
]
HISTORICAL_CPU = 9.19053375000658
HISTORICAL_GPU = 8.267716500005918
KAPPA = "0.00387561167742"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, choices=(3, 6), default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, expected in FIXED.items():
        if hashlib.sha256((SNAPSHOT / "complex" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input hash mismatch {name}")
    report = {"benchmark_commit": "172475c2ca321d3496fda7a2a71478ada5acb948",
              "compressed_input_sha256": FIXED, "kappa": KAPPA,
              "clock": "parent perf_counter immediately around fresh subprocess.run; includes exit",
              "blocks": ORDER[:args.blocks], "start_utc": now(), "runs": []}

    def save():
        (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")

    for index, backend in enumerate(itertools.chain.from_iterable(ORDER[:args.blocks])):
        name = f"{index+1:02d}-{backend}"
        path = out / name
        script = ROOT / "fast_search.py" if backend == "new" else LEGACY / "fused_search.py"
        cmd = [sys.executable, "-I", "-B", str(script),
               "--graph", str(SNAPSHOT / "complex/WORD.json.gz"),
               "--layout", str(SNAPSHOT / "complex/LAYOUT.json.gz"),
               "--kappa", KAPPA, "--out", str(path), "--motifs", "10000"]
        if backend != "new":
            cmd.extend(("--backend", backend))
        started = now()
        tick = time.perf_counter()
        run = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        elapsed = time.perf_counter() - tick
        (out / f"{name}.log").write_text(run.stdout + "\n" + run.stderr, encoding="utf-8")
        item = dict(backend=backend, utc=started, process_wall_s=elapsed,
                    command=cmd, exit_code=run.returncode, report_file=f"{name}/report.json")
        report["runs"].append(item)
        save()
        if run.returncode:
            raise RuntimeError(run.stderr)
        r = json.loads((path / "report.json").read_text())
        expected = ("bd3d941f6528341a92b59fb1226e322e98ab6051d32cc0461b3e7cbcae843c22",
                    {"3": 6117, "4": 25774}, 31891, 10000, 24, 27397025, 1, 1, [])
        actual = tuple(r[k] for k in ("baseline_graph", "total_connected_motifs",
                       "nonexcluded_motifs", "selected_motifs", "options_limit",
                       "assignments_scored", "uncertain_or_positive_assignments_checked_by_exact_histogram",
                       "unique_histograms", "positive_examples"))
        if actual != expected:
            raise AssertionError((name, actual, expected))
        print(f"{name} {elapsed:.4f} s", flush=True)

    medians = {b: statistics.median(i["process_wall_s"] for i in report["runs"]
                                    if i["backend"] == b) for b in ("cpu", "cuda", "new")}
    report.update(medians_s=medians, cpu_speedup=medians["cpu"] / medians["new"],
                  legacy_hybrid_speedup=medians["cuda"] / medians["new"],
                  historical_cpu_speedup=HISTORICAL_CPU / medians["new"],
                  historical_hybrid_speedup=HISTORICAL_GPU / medians["new"],
                  time_target_met=medians["new"] <= 4.60 and medians["cpu"] >= 2*medians["new"],
                  end_utc=now())
    save()
    print(json.dumps({k:v for k,v in report.items() if k in
                      ("medians_s", "cpu_speedup", "legacy_hybrid_speedup", "time_target_met")}))


if __name__ == "__main__":
    main()
