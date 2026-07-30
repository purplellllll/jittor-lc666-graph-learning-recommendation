import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


VARIANTS = [
    {"tag": "v88_d1gate03_d2open", "switch_rate": 0.03, "d2": "open"},
    {"tag": "v89_d1gate05_d2open", "switch_rate": 0.05, "d2": "open"},
    {"tag": "v90_d1gate08_d2open", "switch_rate": 0.08, "d2": "open"},
    {"tag": "v91_d1gate05_d2rr20", "switch_rate": 0.05, "d2": "rr20"},
]


def read_scores(source: Path, name: str) -> np.ndarray:
    if source.is_dir():
        return np.loadtxt(source / name, delimiter=",", dtype=np.float32)
    with zipfile.ZipFile(source) as zf:
        with zf.open(name) as f:
            return np.loadtxt(f, delimiter=",", dtype=np.float32)


def read_bytes(source: Path, name: str) -> bytes:
    if source.is_dir():
        return (source / name).read_bytes()
    with zipfile.ZipFile(source) as zf:
        return zf.read(name)


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    min_v = scores.min(axis=1, keepdims=True)
    max_v = scores.max(axis=1, keepdims=True)
    return (scores - min_v) / np.maximum(max_v - min_v, 1e-12)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1, dtype=np.float32)
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


def write_csv(path: Path, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        for row in scores:
            f.write(",".join(f"{float(v):.8f}" for v in row))
            f.write("\n")


def gated_dataset1(ours: np.ndarray, other: np.ndarray, switch_rate: float) -> tuple[np.ndarray, dict]:
    rank_ours = ranks_from_scores(ours)
    rank_other = ranks_from_scores(other)
    top_ours = np.argmax(ours, axis=1)
    top_other = np.argmax(other, axis=1)
    disagree = top_ours != top_other
    delta = confidence(other) - confidence(ours)
    eligible = np.where(disagree)[0]
    switch_count = max(1, int(round(len(ours) * switch_rate)))
    switch_count = min(switch_count, len(eligible))
    chosen = np.zeros(len(ours), dtype=bool)
    if switch_count > 0:
        order = eligible[np.argsort(-delta[eligible])]
        chosen[order[:switch_count]] = True

    row_weight = np.full(len(ours), 0.92, dtype=np.float32)
    row_weight[chosen] = 0.06
    fused = row_weight[:, None] * (1.0 / rank_ours) + (1.0 - row_weight[:, None]) * (1.0 / rank_other)
    fused = normalize_rows(fused.astype(np.float32))
    top_fused = np.argmax(fused, axis=1)
    stats = {
        "switch_rate_requested": switch_rate,
        "switch_rows": int(chosen.sum()),
        "switch_rows_rate": float(chosen.mean()),
        "top1_ours_rate": float(np.mean(top_fused == top_ours)),
        "top1_open_rate": float(np.mean(top_fused == top_other)),
        "top1_neither_rate": float(np.mean((top_fused != top_ours) & (top_fused != top_other))),
        "ours_open_top1_agreement": float(np.mean(top_ours == top_other)),
        "delta_threshold": float(delta[chosen].min()) if chosen.any() else None,
    }
    return fused, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_zip", type=Path, required=True)
    parser.add_argument("--other_zip", type=Path, required=True)
    parser.add_argument("--rr20_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("fusion_v88_gated"))
    parser.add_argument("--zip_prefix", default="result")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    ours_d1 = read_scores(args.ours_zip, "dataset1.csv")
    other_d1 = read_scores(args.other_zip, "dataset1.csv")
    open_d2 = read_bytes(args.other_zip, "dataset2.csv")
    rr20_d2 = read_bytes(args.rr20_dir, "dataset2.csv")

    report = {}
    for variant in VARIANTS:
        tag = variant["tag"]
        out_dir = args.output_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        fused_d1, stats = gated_dataset1(ours_d1, other_d1, float(variant["switch_rate"]))
        write_csv(out_dir / "dataset1.csv", fused_d1)
        d2_bytes = open_d2 if variant["d2"] == "open" else rr20_d2
        (out_dir / "dataset2.csv").write_bytes(d2_bytes)

        result_zip = Path(f"{args.zip_prefix}_{tag}.zip")
        if result_zip.exists():
            result_zip.unlink()
        with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(out_dir / "dataset1.csv", arcname="dataset1.csv")
            zf.write(out_dir / "dataset2.csv", arcname="dataset2.csv")

        report[tag] = {
            "zip": str(result_zip),
            "size_mb": result_zip.stat().st_size / 1024 / 1024,
            "dataset1": stats,
            "dataset2_source": variant["d2"],
        }

    report_path = args.output_root / "gated_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
