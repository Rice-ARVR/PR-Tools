"""
Re-evaluate every finished run under runs/ and emit a side-by-side comparison.

For each run dir runs/<name>/ containing stage2/weights/best.pt, this loads
the checkpoint and runs .val() on the val and test splits of the matching
dataset, then writes:

  runs/comparison.csv      <- machine-readable, one row per (run, split)
  runs/comparison.md       <- markdown table for quick visual comparison

Usage
-----
  uv run python compare_runs.py
"""

from pathlib import Path

from ultralytics import YOLO

HERE      = Path(__file__).parent
RUNS_ROOT = HERE / "runs"
DATA_ROOT = HERE / "data"

RUNS = [
    ("yolo26n_v1", DATA_ROOT / "version_1" / "data_fixed.yaml"),
    ("yolo26s_v1", DATA_ROOT / "version_1" / "data_fixed.yaml"),
    ("yolo26n_v2", DATA_ROOT / "version_2" / "data_fixed.yaml"),
    ("yolo26s_v2", DATA_ROOT / "version_2" / "data_fixed.yaml"),
]

SPLITS = ["val", "test"]

METRIC_KEYS = {
    "mAP50":    "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "precision":"metrics/precision(B)",
    "recall":   "metrics/recall(B)",
}


def evaluate(run_name: str, data_yaml: Path) -> list[dict]:
    best_pt = RUNS_ROOT / run_name / "stage2" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[skip] {run_name}: {best_pt} not found")
        return []

    model = YOLO(str(best_pt))
    rows = []
    for split in SPLITS:
        print(f"\n>>> {run_name} :: {split}")
        res = model.val(
            data     = str(data_yaml),
            split    = split,
            project  = str(RUNS_ROOT / run_name),
            name     = f"compare_{split}",
            exist_ok = True,
            plots    = False,
            verbose  = False,
        )
        rd = res.results_dict
        row = {"run": run_name, "split": split}
        for label, key in METRIC_KEYS.items():
            row[label] = rd.get(key, float("nan"))
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    headers = ["run", "split", *METRIC_KEYS.keys()]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(
            f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h])
            for h in headers
        ))
    path.write_text("\n".join(lines) + "\n")


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = ["run", "split", *METRIC_KEYS.keys()]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = [
            f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h])
            for h in headers
        ]
        out.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    all_rows: list[dict] = []
    for run_name, data_yaml in RUNS:
        all_rows.extend(evaluate(run_name, data_yaml))

    if not all_rows:
        print("No runs evaluated.")
        return

    csv_path = RUNS_ROOT / "comparison.csv"
    md_path  = RUNS_ROOT / "comparison.md"
    write_csv(all_rows, csv_path)
    write_markdown(all_rows, md_path)

    print("\n" + "=" * 70)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print("=" * 70 + "\n")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
