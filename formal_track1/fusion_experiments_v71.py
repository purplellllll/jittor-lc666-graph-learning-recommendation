import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


DATASETS = ("dataset1.csv", "dataset2.csv")
N_CANDIDATES = 100


VARIANTS = [
    {"tag": "v71_rr_w85_w20", "method": "rr", "w1": 0.85, "w2": 0.20},
    {"tag": "v72_rr_w90_w20", "method": "rr", "w1": 0.90, "w2": 0.20},
    {"tag": "v73_rr_w80_w20", "method": "rr", "w1": 0.80, "w2": 0.20},
    {"tag": "v74_rr_w85_w15", "method": "rr", "w1": 0.85, "w2": 0.15},
    {"tag": "v75_borda_w85_w25", "method": "borda", "w1": 0.85, "w2": 0.25},
    {"tag": "v76_rrf20_w85_w25", "method": "rrf", "w1": 0.85, "w2": 0.25, "k": 20.0},
    {"tag": "v77_rprod_w85_w25", "method": "rank_product", "w1": 0.85, "w2": 0.25},
    {"tag": "v78_adapt_w85_w25_s20", "method": "adaptive_rr", "w1": 0.85, "w2": 0.25, "strength": 0.20},
    {"tag": "v79_adapt_w85_w20_s30", "method": "adaptive_rr", "w1": 0.85, "w2": 0.20, "strength": 0.30},
    {"tag": "v80_adapt_w80_w25_s30", "method": "adaptive_rr", "w1": 0.80, "w2": 0.25, "strength": 0.30},
    {"tag": "v81_rr_w60_w20", "method": "rr", "w1": 0.60, "w2": 0.20},
    {"tag": "v82_norm_w75_w25", "method": "norm", "w1": 0.75, "w2": 0.25},
]


def read_scores(source: Path, name: str) -> np.ndarray:
    if source.is_dir():
        return np.loadtxt(source / name, delimiter=",", dtype=np.float32)
    with zipfile.ZipFile(source) as zf:
        with zf.open(name) as f:
            return np.loadtxt(f, delimiter=",", dtype=np.float32)


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    min_v = scores.min(axis=1, keepdims=True)
    max_v = scores.max(axis=1, keepdims=True)
    denom = np.maximum(max_v - min_v, 1e-12)
    return (scores - min_v) / denom


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1, dtype=np.float32)
    return ranks


def confidence(scores: np.ndarray) -> np.ndarray:
    top2 = np.partition(scores, kth=scores.shape[1] - 2, axis=1)[:, -2:]
    top2.sort(axis=1)
    margin = top2[:, 1] - top2[:, 0]
    spread = np.std(scores, axis=1) + 1e-6
    raw = margin / spread
    med = np.median(raw)
    q25, q75 = np.percentile(raw, [25, 75])
    scale = max(float(q75 - q25), 1e-6)
    z = np.clip((raw - med) / scale, -8.0, 8.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def method_scores(
    variant: dict,
    ours: np.ndarray,
    other: np.ndarray,
    rank_ours: np.ndarray,
    rank_other: np.ndarray,
    weight: float,
    conf_ours: np.ndarray,
    conf_other: np.ndarray,
) -> np.ndarray:
    method = variant["method"]
    if method == "rr":
        fused = weight * (1.0 / rank_ours) + (1.0 - weight) * (1.0 / rank_other)
    elif method == "rrf":
        k = float(variant.get("k", 20.0))
        fused = weight * (1.0 / (k + rank_ours)) + (1.0 - weight) * (1.0 / (k + rank_other))
    elif method == "borda":
        fused = weight * ((N_CANDIDATES + 1.0 - rank_ours) / N_CANDIDATES)
        fused += (1.0 - weight) * ((N_CANDIDATES + 1.0 - rank_other) / N_CANDIDATES)
    elif method == "rank_product":
        fused = -(weight * np.log(rank_ours) + (1.0 - weight) * np.log(rank_other))
    elif method == "norm":
        fused = weight * normalize_rows(ours) + (1.0 - weight) * normalize_rows(other)
    elif method == "adaptive_rr":
        strength = float(variant.get("strength", 0.25))
        row_weight = weight + strength * (conf_ours - conf_other)
        row_weight = np.clip(row_weight, 0.02, 0.98).astype(np.float32)[:, None]
        fused = row_weight * (1.0 / rank_ours) + (1.0 - row_weight) * (1.0 / rank_other)
    else:
        raise ValueError(f"unknown method: {method}")
    return normalize_rows(fused.astype(np.float32))


def write_csv(path: Path, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        for row in scores:
            f.write(",".join(f"{float(v):.8f}" for v in row))
            f.write("\n")


def source_stats(fused: np.ndarray, ours: np.ndarray, other: np.ndarray) -> dict:
    top_fused = np.argmax(fused, axis=1)
    top_ours = np.argmax(ours, axis=1)
    top_other = np.argmax(other, axis=1)
    n = len(top_fused)
    return {
        "rows": int(n),
        "top1_ours_rate": float(np.mean(top_fused == top_ours)),
        "top1_other_rate": float(np.mean(top_fused == top_other)),
        "top1_neither_rate": float(np.mean((top_fused != top_ours) & (top_fused != top_other))),
        "ours_other_top1_agreement": float(np.mean(top_ours == top_other)),
    }


def zip_variant(out_dir: Path, result_zip: Path) -> None:
    if result_zip.exists():
        result_zip.unlink()
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in DATASETS:
            zf.write(out_dir / name, arcname=name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("fusion_v71_experiments"))
    parser.add_argument("--zip_prefix", default="result")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {}

    for name in DATASETS:
        ours = read_scores(args.ours, name)
        other = read_scores(args.other, name)
        rank_ours = ranks_from_scores(ours)
        rank_other = ranks_from_scores(other)
        conf_ours = confidence(ours)
        conf_other = confidence(other)
        dataset_report = {}

        for variant in VARIANTS:
            weight = float(variant["w1"] if name == "dataset1.csv" else variant["w2"])
            fused = method_scores(variant, ours, other, rank_ours, rank_other, weight, conf_ours, conf_other)
            out_dir = args.output_root / variant["tag"]
            write_csv(out_dir / name, fused)
            dataset_report[variant["tag"]] = source_stats(fused, ours, other)
            dataset_report[variant["tag"]]["weight"] = weight
            dataset_report[variant["tag"]]["method"] = variant["method"]

        report[name] = dataset_report

    zip_report = {}
    for variant in VARIANTS:
        tag = variant["tag"]
        out_dir = args.output_root / tag
        result_zip = Path(f"{args.zip_prefix}_{tag}.zip")
        zip_variant(out_dir, result_zip)
        zip_report[tag] = {
            "zip": str(result_zip),
            "size_mb": result_zip.stat().st_size / 1024 / 1024,
        }

    report["zips"] = zip_report
    report_path = args.output_root / "fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
