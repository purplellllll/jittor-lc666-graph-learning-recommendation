"""Build the audited high-score submission from two prediction branches.

The primary branch is produced by ``code/main.py``.  The complementary
branch is a separately archived 100-candidate scorer.  The final audited
preset keeps dataset1 from the primary branch and dataset2 from the
complementary branch.  CSV bytes are preserved exactly while every row is
validated before the result archive is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator


DATASETS = ("dataset1.csv", "dataset2.csv")
EXPECTED_ROWS = {"dataset1.csv": 61051, "dataset2.csv": 153420}
EXPECTED_COLUMNS = 100
EXPECTED_COMPLEMENTARY_ARCHIVE_SHA256 = (
    "974582aa2f8f156bbe7f539487450e3e1d7ca75cfaec6027ee2999ad53fdb461"
)
EXPECTED_PRIMARY_DATASET1_SHA256 = (
    "829b6877d8a4663f512005e4d1d431d906991c517bb3708e03a99b161cdb7ee5"
)
EXPECTED_COMPLEMENTARY_DATASET2_SHA256 = (
    "a56c257f97710bbe8714fcd6d23b1fb4e5b7ea528d2404011457d452f044fb4a"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def open_member(source: Path, member: str) -> Iterator[BinaryIO]:
    """Open one prediction CSV from a directory or ZIP archive."""
    if source.is_dir():
        with (source / member).open("rb") as stream:
            yield stream
        return

    with zipfile.ZipFile(source, "r") as archive:
        if member not in archive.namelist():
            raise ValueError(f"{source} does not contain {member} at archive root")
        with archive.open(member, "r") as stream:
            yield stream


def validate_zip(source: Path) -> None:
    if source.is_dir():
        return
    if not zipfile.is_zipfile(source):
        raise ValueError(f"not a ZIP archive: {source}")
    with zipfile.ZipFile(source, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC check failed for {source}: {bad_member}")


def copy_validated_csv(
    source: Path,
    member: str,
    destination: Path,
    expected_rows: int,
    expected_columns: int,
) -> dict:
    """Copy a CSV byte-for-byte and validate its complete numeric matrix."""
    digest = hashlib.sha256()
    row_count = 0
    byte_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open_member(source, member) as src, destination.open("wb") as dst:
        for row_count, raw_line in enumerate(src, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            dst.write(raw_line)

            try:
                fields = raw_line.decode("ascii").strip().split(",")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{member} row {row_count} is not ASCII") from exc
            if len(fields) != expected_columns:
                raise ValueError(
                    f"{member} row {row_count}: expected {expected_columns} values, "
                    f"found {len(fields)}"
                )
            try:
                values = (float(value) for value in fields)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("non-finite value")
            except ValueError as exc:
                raise ValueError(f"{member} row {row_count} contains an invalid score") from exc

    if row_count != expected_rows:
        raise ValueError(f"{member}: expected {expected_rows} rows, found {row_count}")

    return {
        "source": str(source),
        "rows": row_count,
        "columns": expected_columns,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the high-score dataset-aware dual-branch submission"
    )
    parser.add_argument(
        "--primary_result",
        type=Path,
        default=Path("outputs/primary_v69.zip"),
        help="V69 primary-branch ZIP or directory",
    )
    parser.add_argument(
        "--complementary_result",
        type=Path,
        default=Path("data/complementary_result.zip"),
        help="archived complementary-branch ZIP or directory",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/high_score"))
    parser.add_argument("--result_zip", type=Path, default=Path("outputs/result.zip"))
    parser.add_argument("--dataset1_rows", type=int, default=EXPECTED_ROWS["dataset1.csv"])
    parser.add_argument("--dataset2_rows", type=int, default=EXPECTED_ROWS["dataset2.csv"])
    parser.add_argument("--columns", type=int, default=EXPECTED_COLUMNS)
    parser.add_argument(
        "--allow_unverified_inputs",
        action="store_true",
        help="allow non-archived branch hashes (intended only for controlled experiments)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.primary_result.exists():
        raise FileNotFoundError(f"primary prediction branch not found: {args.primary_result}")
    if not args.complementary_result.exists():
        raise FileNotFoundError(
            "complementary prediction branch not found: "
            f"{args.complementary_result}; see README.md for the required artifact hash"
        )

    validate_zip(args.primary_result)
    validate_zip(args.complementary_result)
    primary_archive_sha256 = (
        sha256_file(args.primary_result) if args.primary_result.is_file() else None
    )
    complementary_archive_sha256 = (
        sha256_file(args.complementary_result)
        if args.complementary_result.is_file()
        else None
    )
    if (
        not args.allow_unverified_inputs
        and complementary_archive_sha256 is not None
        and complementary_archive_sha256 != EXPECTED_COMPLEMENTARY_ARCHIVE_SHA256
    ):
        raise ValueError(
            "complementary archive hash does not match the audited high-score input: "
            f"{complementary_archive_sha256}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_map = {
        "dataset1.csv": args.primary_result,
        "dataset2.csv": args.complementary_result,
    }
    row_map = {
        "dataset1.csv": args.dataset1_rows,
        "dataset2.csv": args.dataset2_rows,
    }
    members = {}
    for member in DATASETS:
        members[member] = copy_validated_csv(
            source_map[member],
            member,
            args.output_dir / member,
            expected_rows=row_map[member],
            expected_columns=args.columns,
        )

    if not args.allow_unverified_inputs:
        expected_member_hashes = {
            "dataset1.csv": EXPECTED_PRIMARY_DATASET1_SHA256,
            "dataset2.csv": EXPECTED_COMPLEMENTARY_DATASET2_SHA256,
        }
        for member, expected_hash in expected_member_hashes.items():
            actual_hash = members[member]["sha256"]
            if actual_hash != expected_hash:
                raise ValueError(
                    f"{member} hash does not match the audited high-score member: "
                    f"{actual_hash}"
                )

    args.result_zip.parent.mkdir(parents=True, exist_ok=True)
    if args.result_zip.exists():
        args.result_zip.unlink()
    with zipfile.ZipFile(
        args.result_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for member in DATASETS:
            archive.write(args.output_dir / member, arcname=member)

    with zipfile.ZipFile(args.result_zip, "r") as archive:
        bad_member = archive.testzip()
        archive_members = archive.namelist()
    if bad_member is not None or archive_members != list(DATASETS):
        raise RuntimeError(
            f"result archive verification failed: members={archive_members}, bad={bad_member}"
        )

    metadata = {
        "preset": "high_score_v83",
        "strategy": "dataset-aware dual-branch routing",
        "routing": {
            "dataset1.csv": "primary temporal-graph ranker",
            "dataset2.csv": "complementary prediction branch",
        },
        "source_files": {
            "primary": {
                "path": str(args.primary_result),
                "sha256": primary_archive_sha256,
            },
            "complementary": {
                "path": str(args.complementary_result),
                "sha256": complementary_archive_sha256,
            },
        },
        "members": members,
        "result_zip": {
            "path": str(args.result_zip),
            "bytes": args.result_zip.stat().st_size,
            "sha256": sha256_file(args.result_zip),
            "zip_test": "passed",
        },
    }
    metadata_path = args.output_dir / "ensemble_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
