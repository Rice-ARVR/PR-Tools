"""
Train every combination of {version_1, version_2} x {yolo26n, yolo26s}.

Produces four runs under runs/:
  runs/yolo26n_v1/
  runs/yolo26s_v1/
  runs/yolo26n_v2/
  runs/yolo26s_v2/

Usage
-----
  uv run python train_all.py
"""

from pathlib import Path

from train_yolo import train

HERE      = Path(__file__).parent
DATA_ROOT = HERE / "data"

DATASETS = [
    ("v1", DATA_ROOT / "version_1"),
    ("v2", DATA_ROOT / "version_2"),
]

MODELS = ["yolo26n.pt", "yolo26s.pt"]


def main() -> None:
    results = []
    for dataset_tag, dataset_dir in DATASETS:
        for model_file in MODELS:
            model_tag = Path(model_file).stem
            run_name  = f"{model_tag}_{dataset_tag}"

            banner = "#" * 70
            print(f"\n{banner}")
            print(f"# Run: {run_name}")
            print(f"#   dataset: {dataset_dir}")
            print(f"#   model:   {model_file}")
            print(banner + "\n")

            best_pt = train(
                dataset_dir = dataset_dir,
                base_model  = model_file,
                run_name    = run_name,
            )
            results.append((run_name, best_pt))

    sep = "=" * 70
    print(f"\n{sep}")
    print("All trainings complete.")
    print(sep)
    for run_name, best_pt in results:
        print(f"  {run_name:20s} -> {best_pt}")


if __name__ == "__main__":
    main()
