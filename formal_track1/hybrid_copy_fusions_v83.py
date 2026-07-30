import argparse
import json
import zipfile
from pathlib import Path


HYBRIDS = [
    {
        "tag": "v83_d1v69_d2open",
        "dataset1": "ours_zip",
        "dataset2": "other_zip",
    },
    {
        "tag": "v84_d1rr85_d2open",
        "dataset1": "v71_dir",
        "dataset2": "other_zip",
    },
    {
        "tag": "v85_d1v69_d2rr20",
        "dataset1": "ours_zip",
        "dataset2": "v71_dir",
    },
    {
        "tag": "v86_d1rr90_d2open",
        "dataset1": "v72_dir",
        "dataset2": "other_zip",
    },
    {
        "tag": "v87_d1v69_d2rr15",
        "dataset1": "ours_zip",
        "dataset2": "v74_dir",
    },
    {
        "tag": "v92_d1borda_d2open",
        "dataset1": "v75_dir",
        "dataset2": "other_zip",
    },
    {
        "tag": "v93_d1rrf20_d2open",
        "dataset1": "v76_dir",
        "dataset2": "other_zip",
    },
    {
        "tag": "v94_d1norm_d2open",
        "dataset1": "v82_dir",
        "dataset2": "other_zip",
    },
]


def read_bytes(source: Path, member: str) -> bytes:
    if source.is_dir():
        return (source / member).read_bytes()
    with zipfile.ZipFile(source) as zf:
        return zf.read(member)


def count_lines(data: bytes) -> int:
    return len(data.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_zip", type=Path, required=True)
    parser.add_argument("--other_zip", type=Path, required=True)
    parser.add_argument("--v71_dir", type=Path, required=True)
    parser.add_argument("--v72_dir", type=Path, required=True)
    parser.add_argument("--v74_dir", type=Path, required=True)
    parser.add_argument("--v75_dir", type=Path)
    parser.add_argument("--v76_dir", type=Path)
    parser.add_argument("--v82_dir", type=Path)
    parser.add_argument("--output_root", type=Path, default=Path("fusion_v83_hybrids"))
    parser.add_argument("--zip_prefix", default="result")
    args = parser.parse_args()

    sources = {
        "ours_zip": args.ours_zip,
        "other_zip": args.other_zip,
        "v71_dir": args.v71_dir,
        "v72_dir": args.v72_dir,
        "v74_dir": args.v74_dir,
        "v75_dir": args.v75_dir,
        "v76_dir": args.v76_dir,
        "v82_dir": args.v82_dir,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {}

    for hybrid in HYBRIDS:
        tag = hybrid["tag"]
        out_dir = args.output_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        result_zip = Path(f"{args.zip_prefix}_{tag}.zip")
        if result_zip.exists():
            result_zip.unlink()

        rows = {}
        with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for dataset in ("dataset1", "dataset2"):
                name = f"{dataset}.csv"
                source = sources[hybrid[dataset]]
                if source is None:
                    raise ValueError(f"missing source argument for {hybrid[dataset]}")
                data = read_bytes(source, name)
                (out_dir / name).write_bytes(data)
                zf.writestr(name, data)
                rows[name] = count_lines(data)

        report[tag] = {
            "zip": str(result_zip),
            "size_mb": result_zip.stat().st_size / 1024 / 1024,
            "rows": rows,
            "sources": {"dataset1": hybrid["dataset1"], "dataset2": hybrid["dataset2"]},
        }

    report_path = args.output_root / "hybrid_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
