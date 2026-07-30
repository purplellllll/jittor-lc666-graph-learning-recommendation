import argparse
import zipfile
from pathlib import Path

import numpy as np


DATASETS = ("dataset1.csv", "dataset2.csv")


def read_scores(source: Path, name: str) -> np.ndarray:
    if source.is_dir():
        return np.loadtxt(source / name, delimiter=",", dtype=np.float32)
    with zipfile.ZipFile(source) as zf:
        with zf.open(name) as f:
            return np.loadtxt(f, delimiter=",", dtype=np.float32)


def reciprocal_rank_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    rank_values = np.arange(1, scores.shape[1] + 1, dtype=np.float32)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = rank_values
    return 1.0 / ranks


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    min_v = scores.min(axis=1, keepdims=True)
    max_v = scores.max(axis=1, keepdims=True)
    denom = np.maximum(max_v - min_v, 1e-12)
    return (scores - min_v) / denom


def write_csv(path: Path, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        for row in scores:
            f.write(",".join(f"{float(v):.8f}" for v in row))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--result_zip", type=Path, required=True)
    parser.add_argument("--w1", type=float, default=0.70, help="ours weight for dataset1")
    parser.add_argument("--w2", type=float, default=0.50, help="ours weight for dataset2")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, weight in zip(DATASETS, (args.w1, args.w2)):
        ours = reciprocal_rank_scores(read_scores(args.ours, name))
        other = reciprocal_rank_scores(read_scores(args.other, name))
        fused = normalize_rows(weight * ours + (1.0 - weight) * other)
        write_csv(args.output_dir / name, fused)

    if args.result_zip.exists():
        args.result_zip.unlink()
    with zipfile.ZipFile(args.result_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in DATASETS:
            zf.write(args.output_dir / name, arcname=name)


if __name__ == "__main__":
    main()
