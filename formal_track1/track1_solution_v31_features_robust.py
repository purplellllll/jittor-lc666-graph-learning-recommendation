import argparse
import json
import math
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_URL = "https://cloud.tsinghua.edu.cn/f/6a9569def9044d49bb96/?dl=1"
CAND_COLS = [f"c{i}" for i in range(1, 101)]


@dataclass
class History:
    base: int
    node_size: int
    pair_keys: np.ndarray
    pair_counts: np.ndarray
    pair_last: np.ndarray
    trans_keys: np.ndarray
    trans_counts: np.ndarray
    trans_last: np.ndarray
    skip_keys: np.ndarray
    skip_counts: np.ndarray
    skip_last: np.ndarray
    dst_count: np.ndarray
    dst_recent_count: np.ndarray
    dst_very_recent_count: np.ndarray
    dst_window_counts: list
    dst_last: np.ndarray
    src_count: np.ndarray
    src_last: np.ndarray
    recent_by_src: np.ndarray
    max_time: int
    min_time: int
    tau: float
    test_cand_count: np.ndarray
    test_src_cand_keys: np.ndarray
    test_src_cand_counts: np.ndarray


def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def ensure_data(data_dir: Path, zip_path: Path | None, url: str):
    if (data_dir / "dataset1" / "train.csv").exists() and (data_dir / "dataset2" / "test.csv").exists():
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = zip_path or Path("track1_data.zip")
    if not archive.exists():
        log(f"Downloading official data to {archive}")
        urllib.request.urlretrieve(url, archive)

    log(f"Extracting {archive} to {data_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(data_dir)
    return data_dir


def read_dataset(data_dir: Path, name: str):
    train = pd.read_csv(data_dir / name / "train.csv")
    test = pd.read_csv(data_dir / name / "test.csv")
    train = train.sort_values("time", kind="mergesort").reset_index(drop=True)
    return train, test


def dense_counts(ids: np.ndarray, counts: np.ndarray, size: int, dtype=np.float32):
    arr = np.zeros(size, dtype=dtype)
    valid = (ids >= 0) & (ids < size)
    arr[ids[valid].astype(np.int64)] = counts[valid].astype(dtype)
    return arr


def dense_last(ids: np.ndarray, values: np.ndarray, size: int):
    arr = np.full(size, -1, dtype=np.int64)
    valid = (ids >= 0) & (ids < size)
    arr[ids[valid].astype(np.int64)] = values[valid].astype(np.int64)
    return arr


def make_recent_by_src(train: pd.DataFrame, node_size: int, k: int):
    recent = np.full((node_size, k), -1, dtype=np.int32)
    src = train["src"].to_numpy(np.int64, copy=False)
    dst = train["dst"].to_numpy(np.int32, copy=False)
    for s, d in zip(src, dst):
        if 0 <= s < node_size:
            recent[s, 1:] = recent[s, :-1]
            recent[s, 0] = d
    return recent


def grouped_pair_arrays(df: pd.DataFrame, base: int):
    if len(df) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    grouped = (
        df.groupby(["src", "dst"], sort=False)
        .agg(cnt=("time", "size"), last=("time", "max"))
        .reset_index()
    )
    keys = grouped["src"].to_numpy(np.int64) * base + grouped["dst"].to_numpy(np.int64)
    order = np.argsort(keys, kind="mergesort")
    return (
        keys[order].astype(np.int64, copy=False),
        grouped["cnt"].to_numpy(np.float32)[order],
        grouped["last"].to_numpy(np.int64)[order],
    )


def grouped_transition_arrays(train: pd.DataFrame, base: int):
    if len(train) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    ordered = train[["src", "dst", "time"]].sort_values(["src", "time"], kind="mergesort")
    prev_dst = ordered.groupby("src", sort=False)["dst"].shift(1)
    trans = ordered.loc[prev_dst.notna(), ["dst", "time"]].copy()
    trans["prev"] = prev_dst[prev_dst.notna()].astype(np.int64).to_numpy()
    grouped = (
        trans.groupby(["prev", "dst"], sort=False)
        .agg(cnt=("time", "size"), last=("time", "max"))
        .reset_index()
    )
    keys = grouped["prev"].to_numpy(np.int64) * base + grouped["dst"].to_numpy(np.int64)
    order = np.argsort(keys, kind="mergesort")
    return (
        keys[order].astype(np.int64, copy=False),
        grouped["cnt"].to_numpy(np.float32)[order],
        grouped["last"].to_numpy(np.int64)[order],
    )


def grouped_skip_arrays(train: pd.DataFrame, base: int, max_lag: int = 5):
    if len(train) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    ordered = train[["src", "dst", "time"]].sort_values(["src", "time"], kind="mergesort")
    src = ordered["src"].to_numpy(np.int64, copy=False)
    dst = ordered["dst"].to_numpy(np.int64, copy=False)
    times = ordered["time"].to_numpy(np.int64, copy=False)
    key_parts = []
    time_parts = []
    for lag in range(1, max_lag + 1):
        if len(src) <= lag:
            break
        same_src = src[lag:] == src[:-lag]
        if not np.any(same_src):
            continue
        prev = dst[:-lag][same_src]
        cur = dst[lag:][same_src]
        key_parts.append(prev * base + cur)
        time_parts.append(times[lag:][same_src])
    if not key_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    keys = np.concatenate(key_parts).astype(np.int64, copy=False)
    last_times = np.concatenate(time_parts).astype(np.int64, copy=False)
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    times_sorted = last_times[order]
    unique_keys, first_idx, counts = np.unique(keys_sorted, return_index=True, return_counts=True)
    max_last = np.maximum.reduceat(times_sorted, first_idx)
    return unique_keys, counts.astype(np.float32), max_last.astype(np.int64)


def build_history(train: pd.DataFrame, test: pd.DataFrame, recent_k: int = 5):
    cand = test[CAND_COLS].to_numpy(np.int64, copy=False)
    max_id = int(
        max(
            train["src"].max(),
            train["dst"].max(),
            test["src"].max(),
            cand.max(),
        )
    )
    node_size = max_id + 2
    base = node_size
    min_time = int(train["time"].min())
    max_time = int(train["time"].max())
    tau = max((max_time - min_time) / 10.0, 1.0)

    log("  building pair aggregates")
    pair_keys, pair_counts, pair_last = grouped_pair_arrays(train, base)

    log("  building transition aggregates")
    trans_keys, trans_counts, trans_last = grouped_transition_arrays(train, base)

    log("  building skip-transition aggregates")
    skip_keys, skip_counts, skip_last = grouped_skip_arrays(train, base, max_lag=5)

    log("  building node aggregates")
    dst_group = train.groupby("dst", sort=False).agg(cnt=("time", "size"), last=("time", "max")).reset_index()
    dst_count = dense_counts(dst_group["dst"].to_numpy(np.int64), dst_group["cnt"].to_numpy(np.float32), node_size)
    dst_last = dense_last(dst_group["dst"].to_numpy(np.int64), dst_group["last"].to_numpy(np.int64), node_size)

    train_times = train["time"].to_numpy(np.int64)
    q_recent = np.quantile(train_times, 0.85)
    q_very_recent = np.quantile(train_times, 0.97)
    recent_group = (
        train.loc[train["time"] >= q_recent]
        .groupby("dst", sort=False)
        .size()
        .reset_index(name="cnt")
    )
    very_recent_group = (
        train.loc[train["time"] >= q_very_recent]
        .groupby("dst", sort=False)
        .size()
        .reset_index(name="cnt")
    )
    dst_recent_count = dense_counts(
        recent_group["dst"].to_numpy(np.int64), recent_group["cnt"].to_numpy(np.float32), node_size
    )
    dst_very_recent_count = dense_counts(
        very_recent_group["dst"].to_numpy(np.int64), very_recent_group["cnt"].to_numpy(np.float32), node_size
    )
    dst_window_counts = []
    for q in [0.50, 0.70, 0.85, 0.90, 0.95, 0.97, 0.99]:
        threshold = np.quantile(train_times, q)
        group = train.loc[train["time"] >= threshold].groupby("dst", sort=False).size().reset_index(name="cnt")
        dst_window_counts.append(
            dense_counts(group["dst"].to_numpy(np.int64), group["cnt"].to_numpy(np.float32), node_size)
        )

    src_group = train.groupby("src", sort=False).agg(cnt=("time", "size"), last=("time", "max")).reset_index()
    src_count = dense_counts(src_group["src"].to_numpy(np.int64), src_group["cnt"].to_numpy(np.float32), node_size)
    src_last = dense_last(src_group["src"].to_numpy(np.int64), src_group["last"].to_numpy(np.int64), node_size)

    log("  building recent source histories")
    recent_by_src = make_recent_by_src(train, node_size, recent_k)

    flat_cand = cand.reshape(-1)
    test_ids, test_counts = np.unique(flat_cand, return_counts=True)
    test_cand_count = dense_counts(test_ids, test_counts.astype(np.float32), node_size)
    test_src_rep = np.repeat(test["src"].to_numpy(np.int64, copy=False), cand.shape[1])
    test_src_cand_raw = test_src_rep * base + flat_cand
    test_src_cand_keys, test_src_cand_counts = np.unique(test_src_cand_raw, return_counts=True)

    return History(
        base=base,
        node_size=node_size,
        pair_keys=pair_keys,
        pair_counts=pair_counts,
        pair_last=pair_last,
        trans_keys=trans_keys,
        trans_counts=trans_counts,
        trans_last=trans_last,
        skip_keys=skip_keys,
        skip_counts=skip_counts,
        skip_last=skip_last,
        dst_count=dst_count,
        dst_recent_count=dst_recent_count,
        dst_very_recent_count=dst_very_recent_count,
        dst_window_counts=dst_window_counts,
        dst_last=dst_last,
        src_count=src_count,
        src_last=src_last,
        recent_by_src=recent_by_src,
        max_time=max_time,
        min_time=min_time,
        tau=tau,
        test_cand_count=test_cand_count,
        test_src_cand_keys=test_src_cand_keys.astype(np.int64, copy=False),
        test_src_cand_counts=test_src_cand_counts.astype(np.float32, copy=False),
    )


def lookup(keys_sorted, values, query_keys, default=0.0):
    out = np.full(query_keys.shape, default, dtype=np.float32)
    if len(keys_sorted) == 0:
        return out
    idx = np.searchsorted(keys_sorted, query_keys)
    valid = idx < len(keys_sorted)
    mask = np.zeros(query_keys.shape, dtype=bool)
    mask[valid] = keys_sorted[idx[valid]] == query_keys[valid]
    out[mask] = values[idx[mask]]
    return out


def lookup_last(keys_sorted, values, query_keys):
    out = np.full(query_keys.shape, -1, dtype=np.int64)
    if len(keys_sorted) == 0:
        return out
    idx = np.searchsorted(keys_sorted, query_keys)
    valid = idx < len(keys_sorted)
    mask = np.zeros(query_keys.shape, dtype=bool)
    mask[valid] = keys_sorted[idx[valid]] == query_keys[valid]
    out[mask] = values[idx[mask]]
    return out


def hash_noise(candidates):
    x = candidates.astype(np.uint64, copy=False)
    x = (x ^ np.uint64(0x9E3779B97F4A7C15)) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0x94D049BB133111EB)
    return ((x ^ (x >> np.uint64(31))) & np.uint64(0xFFFF)).astype(np.float32) / 65535.0


def score_batch(history: History, src, times, candidates, weights):
    n, m = candidates.shape
    cand_flat = candidates.reshape(-1).astype(np.int64, copy=False)
    src_rep = np.repeat(src.astype(np.int64, copy=False), m)
    time_rep = np.repeat(times.astype(np.int64, copy=False), m)
    pair_query = src_rep * history.base + cand_flat

    pair_cnt = lookup(history.pair_keys, history.pair_counts, pair_query).reshape(n, m)
    pair_last = lookup_last(history.pair_keys, history.pair_last, pair_query).reshape(n, m)
    pair_delta = np.maximum(times[:, None].astype(np.int64) - pair_last, 0)
    pair_rec = np.where(pair_last >= 0, np.exp(-pair_delta / history.tau), 0.0).astype(np.float32)

    scores = (
        weights["pair_count"] * np.log1p(pair_cnt)
        + weights["pair_recency"] * pair_rec
    ).astype(np.float32)

    valid_cand = np.clip(candidates.astype(np.int64, copy=False), 0, history.node_size - 1)
    dst_cnt = history.dst_count[valid_cand]
    dst_recent = history.dst_recent_count[valid_cand]
    dst_very_recent = history.dst_very_recent_count[valid_cand]
    dst_last = history.dst_last[valid_cand]
    dst_delta = np.maximum(times[:, None].astype(np.int64) - dst_last, 0)
    dst_rec = np.where(dst_last >= 0, np.exp(-dst_delta / history.tau), 0.0).astype(np.float32)
    test_freq = history.test_cand_count[valid_cand]

    scores += weights["dst_pop"] * np.log1p(dst_cnt)
    scores += weights["dst_recent"] * np.log1p(dst_recent)
    scores += weights["dst_very_recent"] * np.log1p(dst_very_recent)
    scores += weights["dst_recency"] * dst_rec
    scores += weights["test_freq"] * np.log1p(test_freq)

    src_safe = np.clip(src.astype(np.int64, copy=False), 0, history.node_size - 1)
    recent = history.recent_by_src[src_safe]
    recent_weights = weights["recent_match"]
    trans_weights = weights["transition"]
    trans_rec_weights = weights["transition_recency"]
    skip_weights = weights["skip_transition"]
    skip_rec_weights = weights["skip_transition_recency"]
    for j in range(min(recent.shape[1], len(recent_weights))):
        prev = recent[:, j].astype(np.int64)
        has_prev = prev >= 0
        if not np.any(has_prev):
            continue
        scores += recent_weights[j] * (candidates == prev[:, None]).astype(np.float32)

        trans_query = np.repeat(np.maximum(prev, 0), m) * history.base + cand_flat
        trans_cnt = lookup(history.trans_keys, history.trans_counts, trans_query).reshape(n, m)
        trans_last = lookup_last(history.trans_keys, history.trans_last, trans_query).reshape(n, m)
        trans_delta = np.maximum(times[:, None].astype(np.int64) - trans_last, 0)
        trans_rec = np.where(trans_last >= 0, np.exp(-trans_delta / history.tau), 0.0).astype(np.float32)
        scores += trans_weights[j] * np.log1p(trans_cnt)
        scores += trans_rec_weights[j] * trans_rec

        skip_cnt = lookup(history.skip_keys, history.skip_counts, trans_query).reshape(n, m)
        skip_last = lookup_last(history.skip_keys, history.skip_last, trans_query).reshape(n, m)
        skip_delta = np.maximum(times[:, None].astype(np.int64) - skip_last, 0)
        skip_rec = np.where(skip_last >= 0, np.exp(-skip_delta / history.tau), 0.0).astype(np.float32)
        scores += skip_weights[j] * np.log1p(skip_cnt)
        scores += skip_rec_weights[j] * skip_rec

    scores += weights["noise"] * hash_noise(candidates)
    return scores


FEATURE_WINDOW_QS = [0.50, 0.70, 0.85, 0.90, 0.95, 0.97, 0.99]


def ranker_feature_names():
    names = [
        "log_pair_cnt",
        "pair_rec",
        "log_rev_pair_cnt",
        "rev_pair_rec",
        "log_dst_cnt",
        "log_dst_recent85",
        "log_dst_recent97",
        "dst_rec",
        "log_test_freq",
        "log_src_cnt",
        "src_rec",
        "log_cand_src_cnt",
        "cand_src_rec",
        "time_norm",
    ]
    names.extend([f"log_dst_window_{int(q * 100):02d}" for q in FEATURE_WINDOW_QS])
    names.extend(
        [
            "log_test_src_cand_freq",
            "test_src_cand_row_norm",
            "test_src_cand_inv_rank",
            "dst_count_row_norm",
            "dst_recent_row_norm",
            "dst_very_recent_row_norm",
            "test_freq_row_norm",
            "cand_src_count_row_norm",
            "dst_count_inv_rank",
            "dst_recent_inv_rank",
            "dst_very_recent_inv_rank",
            "test_freq_inv_rank",
            "cand_src_count_inv_rank",
        ]
    )
    for j in range(5):
        names.append(f"recent_match_{j}")
    for j in range(5):
        names.append(f"log_trans_cnt_{j}")
        names.append(f"trans_rec_{j}")
        names.append(f"log_skip_cnt_{j}")
        names.append(f"skip_rec_{j}")
    names.extend(
        [
            "v18_score",
            "v18_norm",
            "v18_inv_rank",
            "v18_zscore",
            "v18_margin_best",
            "pos_norm",
            "inv_pos",
        ]
    )
    return names


RANKER_FEATURE_NAMES = ranker_feature_names()


def inverse_rank_feature(scores):
    n, m = scores.shape
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranks = np.empty((n, m), dtype=np.float32)
    ranks[np.arange(n)[:, None], order] = np.arange(m, dtype=np.float32)[None, :]
    return 1.0 / (ranks + 1.0)


def zscore_rows(scores):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / (std + 1e-6)


def feature_batch(history: History, src, times, candidates, base_score_weights=None):
    n, m = candidates.shape
    cand_flat = candidates.reshape(-1).astype(np.int64, copy=False)
    src_int = src.astype(np.int64, copy=False)
    times_int = times.astype(np.int64, copy=False)
    src_rep = np.repeat(src_int, m)
    pair_query = src_rep * history.base + cand_flat
    rev_pair_query = cand_flat * history.base + src_rep

    pair_cnt = lookup(history.pair_keys, history.pair_counts, pair_query).reshape(n, m)
    pair_last = lookup_last(history.pair_keys, history.pair_last, pair_query).reshape(n, m)
    pair_delta = np.maximum(times_int[:, None] - pair_last, 0)
    pair_rec = np.where(pair_last >= 0, np.exp(-pair_delta / history.tau), 0.0).astype(np.float32)

    rev_pair_cnt = lookup(history.pair_keys, history.pair_counts, rev_pair_query).reshape(n, m)
    rev_pair_last = lookup_last(history.pair_keys, history.pair_last, rev_pair_query).reshape(n, m)
    rev_pair_delta = np.maximum(times_int[:, None] - rev_pair_last, 0)
    rev_pair_rec = np.where(rev_pair_last >= 0, np.exp(-rev_pair_delta / history.tau), 0.0).astype(np.float32)

    valid_cand = np.clip(candidates.astype(np.int64, copy=False), 0, history.node_size - 1)
    src_safe = np.clip(src_int, 0, history.node_size - 1)
    dst_cnt = history.dst_count[valid_cand]
    dst_recent = history.dst_recent_count[valid_cand]
    dst_very_recent = history.dst_very_recent_count[valid_cand]
    dst_last = history.dst_last[valid_cand]
    dst_delta = np.maximum(times_int[:, None] - dst_last, 0)
    dst_rec = np.where(dst_last >= 0, np.exp(-dst_delta / history.tau), 0.0).astype(np.float32)
    test_freq = history.test_cand_count[valid_cand]
    test_src_cand_freq = lookup(history.test_src_cand_keys, history.test_src_cand_counts, pair_query).reshape(n, m)

    src_cnt = history.src_count[src_safe]
    src_last = history.src_last[src_safe]
    src_delta = np.maximum(times_int - src_last, 0)
    src_rec = np.where(src_last >= 0, np.exp(-src_delta / history.tau), 0.0).astype(np.float32)
    cand_src_cnt = history.src_count[valid_cand]
    cand_src_last = history.src_last[valid_cand]
    cand_src_delta = np.maximum(times_int[:, None] - cand_src_last, 0)
    cand_src_rec = np.where(cand_src_last >= 0, np.exp(-cand_src_delta / history.tau), 0.0).astype(np.float32)

    time_span = max(history.max_time - history.min_time, 1)
    time_norm = ((times_int - history.min_time) / time_span).astype(np.float32)

    log_dst_cnt = np.log1p(dst_cnt).astype(np.float32)
    log_dst_recent = np.log1p(dst_recent).astype(np.float32)
    log_dst_very_recent = np.log1p(dst_very_recent).astype(np.float32)
    log_test_freq = np.log1p(test_freq).astype(np.float32)
    log_cand_src_cnt = np.log1p(cand_src_cnt).astype(np.float32)
    log_test_src_cand_freq = np.log1p(test_src_cand_freq).astype(np.float32)

    feats = [
        np.log1p(pair_cnt),
        pair_rec,
        np.log1p(rev_pair_cnt),
        rev_pair_rec,
        log_dst_cnt,
        log_dst_recent,
        log_dst_very_recent,
        dst_rec,
        log_test_freq,
        np.broadcast_to(np.log1p(src_cnt).astype(np.float32)[:, None], (n, m)),
        np.broadcast_to(src_rec[:, None], (n, m)),
        log_cand_src_cnt,
        cand_src_rec,
        np.broadcast_to(time_norm[:, None], (n, m)),
    ]
    for counts in history.dst_window_counts:
        feats.append(np.log1p(counts[valid_cand]))
    feats.extend(
        [
            log_test_src_cand_freq,
            normalize_rows(log_test_src_cand_freq),
            inverse_rank_feature(log_test_src_cand_freq),
            normalize_rows(log_dst_cnt),
            normalize_rows(log_dst_recent),
            normalize_rows(log_dst_very_recent),
            normalize_rows(log_test_freq),
            normalize_rows(log_cand_src_cnt),
            inverse_rank_feature(log_dst_cnt),
            inverse_rank_feature(log_dst_recent),
            inverse_rank_feature(log_dst_very_recent),
            inverse_rank_feature(log_test_freq),
            inverse_rank_feature(log_cand_src_cnt),
        ]
    )

    recent = history.recent_by_src[src_safe]
    for j in range(5):
        if j >= recent.shape[1]:
            feats.append(np.zeros((n, m), dtype=np.float32))
            continue
        prev = recent[:, j].astype(np.int64)
        has_prev = prev >= 0
        feats.append(((candidates == prev[:, None]) & has_prev[:, None]).astype(np.float32))

    for j in range(5):
        if j >= recent.shape[1]:
            feats.extend([np.zeros((n, m), dtype=np.float32) for _ in range(4)])
            continue
        prev = recent[:, j].astype(np.int64)
        has_prev = prev >= 0
        row_mask = has_prev[:, None]
        trans_query = np.repeat(np.maximum(prev, 0), m) * history.base + cand_flat

        trans_cnt = lookup(history.trans_keys, history.trans_counts, trans_query).reshape(n, m)
        trans_last = lookup_last(history.trans_keys, history.trans_last, trans_query).reshape(n, m)
        trans_delta = np.maximum(times_int[:, None] - trans_last, 0)
        trans_rec = np.where(trans_last >= 0, np.exp(-trans_delta / history.tau), 0.0).astype(np.float32)

        skip_cnt = lookup(history.skip_keys, history.skip_counts, trans_query).reshape(n, m)
        skip_last = lookup_last(history.skip_keys, history.skip_last, trans_query).reshape(n, m)
        skip_delta = np.maximum(times_int[:, None] - skip_last, 0)
        skip_rec = np.where(skip_last >= 0, np.exp(-skip_delta / history.tau), 0.0).astype(np.float32)

        feats.extend(
            [
                np.where(row_mask, np.log1p(trans_cnt), 0.0).astype(np.float32),
                np.where(row_mask, trans_rec, 0.0).astype(np.float32),
                np.where(row_mask, np.log1p(skip_cnt), 0.0).astype(np.float32),
                np.where(row_mask, skip_rec, 0.0).astype(np.float32),
            ]
        )

    if base_score_weights is None:
        base_score_weights = fixed_best_dataset2_weights()
    base_scores = score_batch(history, src_int, times_int, candidates, base_score_weights)
    positions = np.broadcast_to(np.arange(m, dtype=np.float32)[None, :], (n, m))
    base_best = base_scores.max(axis=1, keepdims=True)
    feats.extend(
        [
            base_scores.astype(np.float32),
            normalize_rows(base_scores).astype(np.float32),
            inverse_rank_feature(base_scores),
            zscore_rows(base_scores).astype(np.float32),
            (base_scores - base_best).astype(np.float32),
            positions / max(m - 1, 1),
            1.0 / (positions + 1.0),
        ]
    )

    stacked = np.stack([x.astype(np.float32, copy=False) for x in feats], axis=2)
    return stacked.reshape(n * m, stacked.shape[2])


def normalize_rows(scores):
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    return (scores - lo) / (hi - lo + 1e-12)


def mrr_from_scores(scores, true_pos):
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.int32)
    for i in range(len(scores)):
        ranks[i] = int(np.where(order[i] == true_pos[i])[0][0]) + 1
    return float(np.mean(1.0 / ranks))


def make_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
    max_rows: int,
    seed: int,
    candidate_mode: str = "flat",
):
    rng = np.random.default_rng(seed)
    if name == "dataset2" and "split" in train.columns and (train["split"] == 1).any():
        hist = train.loc[train["split"] == 0].copy()
        val = train.loc[train["split"] == 1, ["src", "dst", "time"]].copy()
    else:
        cutoff = int(len(train) * 0.85)
        hist = train.iloc[:cutoff].copy()
        val = train.iloc[cutoff:][["src", "dst", "time"]].copy()
    if len(val) > max_rows:
        val = val.sample(max_rows, random_state=seed).sort_values("time", kind="mergesort")
    official_candidates = test[CAND_COLS].to_numpy(np.int64, copy=False)
    negatives_pool = official_candidates.reshape(-1)
    n = len(val)
    if candidate_mode == "row":
        sampled_rows = rng.integers(0, len(official_candidates), size=n)
        candidates = official_candidates[sampled_rows].copy()
    elif candidate_mode == "hybrid":
        sampled_rows = rng.integers(0, len(official_candidates), size=n)
        candidates = official_candidates[sampled_rows].copy()
        replace_cols = rng.choice(100, size=(n, 35), replace=True)
        random_negatives = rng.choice(negatives_pool, size=(n, 35), replace=True)
        row_idx = np.repeat(np.arange(n), replace_cols.shape[1])
        candidates[row_idx, replace_cols.reshape(-1)] = random_negatives.reshape(-1)
    else:
        candidates = rng.choice(negatives_pool, size=(n, 100), replace=True)
    true_pos = rng.integers(0, 100, size=n)
    candidates[np.arange(n), true_pos] = val["dst"].to_numpy(np.int64)
    fake_test = pd.DataFrame(candidates, columns=CAND_COLS)
    fake_test.insert(0, "time", val["time"].to_numpy(np.int64))
    fake_test.insert(0, "src", val["src"].to_numpy(np.int64))
    return hist, val, fake_test, true_pos


def base_weights(name: str):
    if name == "dataset1":
        return {
            "pair_count": 8.0,
            "pair_recency": 5.0,
            "dst_pop": 0.25,
            "dst_recent": 0.35,
            "dst_very_recent": 0.35,
            "dst_recency": 0.20,
            "test_freq": 0.04,
            "recent_match": [4.5, 3.0, 2.0, 1.2, 0.8],
            "transition": [1.8, 1.0, 0.7, 0.4, 0.2],
            "transition_recency": [1.4, 0.8, 0.5, 0.3, 0.2],
            "skip_transition": [0.6, 0.4, 0.25, 0.15, 0.1],
            "skip_transition_recency": [0.3, 0.2, 0.15, 0.1, 0.05],
            "noise": 1e-5,
        }
    return {
        "pair_count": 2.2,
        "pair_recency": 1.2,
        "dst_pop": 1.2,
        "dst_recent": 0.9,
        "dst_very_recent": 0.8,
        "dst_recency": 0.25,
        "test_freq": 0.18,
        "recent_match": [1.2, 0.9, 0.6, 0.35, 0.2],
        "transition": [3.0, 1.8, 1.0, 0.5, 0.25],
        "transition_recency": [1.5, 0.9, 0.5, 0.25, 0.1],
        "skip_transition": [2.4, 1.5, 0.9, 0.5, 0.25],
        "skip_transition_recency": [1.0, 0.6, 0.35, 0.2, 0.1],
        "noise": 1e-5,
    }


def fixed_best_weights(name: str):
    if name != "dataset1":
        return None
    weights = base_weights("dataset1")
    weights["pair_count"] *= 1.2
    weights["pair_recency"] *= 1.2
    weights["transition"] = [x * 2.2 for x in weights["transition"]]
    weights["transition_recency"] = [x * 2.2 for x in weights["transition_recency"]]
    weights["skip_transition"] = [x * 4.0 for x in weights["skip_transition"]]
    weights["skip_transition_recency"] = [x * 4.0 for x in weights["skip_transition_recency"]]
    weights["dst_pop"] *= 0.10
    weights["dst_recent"] *= 0.10
    weights["dst_very_recent"] *= 0.10
    weights["test_freq"] *= 0.20
    return weights, 0.6726111374246836


def fixed_best_dataset2_weights():
    weights = base_weights("dataset2")
    weights["pair_count"] = 0.0
    weights["pair_recency"] = 0.0
    weights["dst_pop"] = 6.24
    weights["dst_recent"] = 17.784
    weights["dst_very_recent"] = 18.304000000000002
    weights["dst_recency"] = 0.25
    weights["test_freq"] = 3.51
    weights["transition"] = [3.6, 2.16, 1.2, 0.6, 0.3]
    weights["transition_recency"] = [1.8, 1.08, 0.6, 0.3, 0.12]
    weights["skip_transition"] = [7.2, 4.5, 2.7, 1.5, 0.75]
    weights["skip_transition_recency"] = [3.0, 1.8, 1.05, 0.6, 0.3]
    return weights


def candidate_weight_sets(name: str):
    w = base_weights(name)
    if name == "dataset2":
        focused = [(0.00, 1.20, 3.0, 5.2, 19.5, 1.0, 3.80, 4.40)]
        for pop_scale in [5.0, 5.1, 5.2, 5.3, 5.4]:
            for freq_scale in [19.2, 19.5, 19.8]:
                focused.append((0.00, 1.20, 3.0, pop_scale, freq_scale, 1.0, 3.80, 4.40))
        for recent_pop_scale in [3.70, 3.80, 3.90]:
            for very_recent_pop_scale in [4.30, 4.40, 4.55]:
                focused.append(
                    (0.00, 1.20, 3.0, 5.2, 19.5, 1.0, recent_pop_scale, very_recent_pop_scale)
                )
        for pop_scale in [5.1, 5.2, 5.3]:
            for recent_pop_scale in [3.75, 3.85]:
                for very_recent_pop_scale in [4.35, 4.50]:
                    focused.append(
                        (0.00, 1.20, 3.0, pop_scale, 19.5, 1.0, recent_pop_scale, very_recent_pop_scale)
                    )
        for skip_scale in [2.9, 3.0, 3.1]:
            focused.append((0.00, 1.20, skip_scale, 5.2, 19.5, 1.0, 3.80, 4.40))
        for dst_recency_scale in [0.0, 0.5, 1.5, 2.0]:
            focused.append((0.00, 1.20, 3.0, 5.2, 19.5, 1.0, 3.80, 4.40, dst_recency_scale))

        deduped = []
        seen = set()
        for spec in focused:
            key = tuple(round(float(x), 6) for x in spec)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(spec)
        focused = deduped

        variants = []
        for spec in focused:
            pair_scale, trans_scale, skip_scale, pop_scale, freq_scale, *extras = spec
            recent_scale = extras[0] if len(extras) > 0 else 1.0
            recent_pop_scale = extras[1] if len(extras) > 1 else 1.0
            very_recent_pop_scale = extras[2] if len(extras) > 2 else recent_pop_scale
            dst_recency_scale = extras[3] if len(extras) > 3 else 1.0
            v = json.loads(json.dumps(w))
            v["pair_count"] *= pair_scale
            v["pair_recency"] *= pair_scale
            v["dst_recency"] *= dst_recency_scale
            v["recent_match"] = [x * recent_scale for x in v["recent_match"]]
            v["transition"] = [x * trans_scale for x in v["transition"]]
            v["transition_recency"] = [x * trans_scale for x in v["transition_recency"]]
            v["skip_transition"] = [x * skip_scale for x in v["skip_transition"]]
            v["skip_transition_recency"] = [x * skip_scale for x in v["skip_transition_recency"]]
            v["dst_pop"] *= pop_scale
            v["dst_recent"] *= pop_scale * recent_pop_scale
            v["dst_very_recent"] *= pop_scale * very_recent_pop_scale
            v["test_freq"] *= freq_scale
            variants.append(v)
        return variants
    variants = [w]
    shared = [
        (1.2, 0.8, 0.8, 0.8, 1.0),
        (0.8, 1.2, 1.1, 1.0, 1.0),
        (0.7, 1.4, 1.4, 1.2, 1.4),
        (1.0, 0.6, 1.6, 1.5, 1.8),
        (1.5, 0.5, 0.6, 0.6, 0.8),
        (0.6, 0.9, 2.0, 1.3, 1.6),
    ]
    if name == "dataset1":
        shared.extend(
            [
                (0.65, 1.5, 1.4, 0.9, 1.0),
                (0.55, 1.8, 1.8, 0.8, 1.1),
                (0.85, 1.6, 1.7, 0.7, 0.9),
                (0.45, 2.0, 2.2, 1.0, 1.2),
                (0.75, 1.1, 2.5, 1.1, 1.5),
                (0.80, 1.9, 2.1, 0.55, 0.8),
                (0.90, 2.0, 2.0, 0.45, 0.7),
                (0.75, 2.3, 2.6, 0.60, 0.9),
                (0.95, 1.8, 2.8, 0.50, 0.6),
                (0.70, 2.6, 3.0, 0.70, 1.0),
                (1.00, 2.2, 3.2, 0.35, 0.5),
                (1.10, 2.0, 3.0, 0.25, 0.4),
                (0.95, 2.5, 3.5, 0.20, 0.3),
                (1.20, 2.2, 4.0, 0.10, 0.2),
                (0.90, 3.0, 4.0, 0.00, 0.1),
                (1.30, 2.4, 4.5, 0.00, 0.05),
                (1.40, 2.0, 5.0, 0.00, 0.00),
                (1.20, 3.0, 5.0, 0.00, 0.10),
                (1.50, 2.5, 6.0, 0.00, 0.00),
                (1.00, 3.5, 6.0, 0.00, 0.00),
            ]
        )
    else:
        shared.extend(
            [
                (0.5, 0.8, 2.5, 1.4, 1.8),
                (0.4, 0.7, 3.0, 1.3, 2.0),
                (0.6, 0.5, 3.5, 1.6, 2.2),
                (0.3, 0.4, 4.0, 1.8, 2.5),
                (0.8, 0.8, 2.5, 1.0, 1.2),
                (0.6, 1.2, 2.5, 1.2, 1.4),
                (0.5, 1.0, 3.0, 1.1, 1.6),
                (0.7, 0.6, 3.0, 1.5, 2.0),
                (0.25, 0.25, 5.0, 2.0, 3.0),
                (0.20, 0.20, 6.0, 2.2, 3.5),
                (0.35, 0.30, 5.0, 2.0, 3.2),
                (0.25, 0.50, 4.5, 2.3, 3.0),
                (0.15, 0.15, 7.0, 2.0, 4.0),
                (0.40, 0.20, 6.0, 1.7, 2.8),
                (0.20, 0.60, 4.8, 2.5, 3.5),
                (0.25, 0.70, 5.5, 2.5, 4.0),
                (0.15, 0.50, 5.5, 2.8, 4.0),
                (0.30, 0.40, 4.0, 3.0, 4.5),
                (0.25, 0.80, 3.5, 3.0, 5.0),
                (0.05, 0.50, 6.0, 3.0, 5.0),
                (0.25, 1.00, 3.2, 3.5, 6.0),
                (0.20, 1.00, 3.0, 4.0, 7.0),
                (0.15, 0.80, 3.5, 4.0, 6.0),
                (0.30, 1.20, 2.5, 3.5, 8.0),
                (0.25, 0.80, 2.0, 5.0, 10.0),
                (0.00, 1.00, 3.5, 4.0, 8.0),
                (0.00, 1.00, 3.0, 5.0, 10.0),
                (0.00, 1.20, 3.0, 5.0, 10.0),
                (0.00, 0.80, 3.0, 6.0, 12.0),
                (0.00, 1.00, 2.0, 6.0, 12.0),
                (0.00, 1.50, 2.5, 5.0, 10.0),
                (0.00, 1.00, 4.0, 5.0, 8.0),
                (0.00, 0.70, 4.0, 5.5, 10.0),
                (0.00, 1.30, 3.5, 4.5, 12.0),
                (0.05, 1.00, 3.0, 5.0, 10.0),
                (0.00, 0.60, 5.0, 5.0, 12.0),
                (0.00, 1.60, 2.0, 6.0, 10.0),
                (0.00, 1.00, 3.5, 6.0, 14.0),
                (0.00, 1.20, 3.5, 5.0, 13.0),
                (0.00, 1.30, 3.5, 5.0, 14.0),
                (0.00, 1.40, 3.5, 5.0, 13.0),
                (0.00, 1.30, 3.0, 5.0, 13.0),
                (0.00, 1.30, 4.0, 5.0, 13.0),
                (0.00, 1.20, 3.0, 5.5, 14.0),
                (0.00, 1.20, 4.0, 5.5, 14.0),
                (0.00, 1.40, 3.5, 5.5, 14.0),
                (0.00, 1.50, 3.5, 5.0, 12.0),
                (0.00, 1.10, 3.5, 5.5, 15.0),
                (0.00, 1.30, 3.5, 6.0, 16.0),
                (0.00, 1.40, 3.0, 5.5, 14.0),
            ]
        )
    for pair_scale, trans_scale, skip_scale, pop_scale, freq_scale in shared:
        v = json.loads(json.dumps(w))
        v["pair_count"] *= pair_scale
        v["pair_recency"] *= pair_scale
        v["transition"] = [x * trans_scale for x in v["transition"]]
        v["transition_recency"] = [x * trans_scale for x in v["transition_recency"]]
        v["skip_transition"] = [x * skip_scale for x in v["skip_transition"]]
        v["skip_transition_recency"] = [x * skip_scale for x in v["skip_transition_recency"]]
        v["dst_pop"] *= pop_scale
        v["dst_recent"] *= pop_scale
        v["dst_very_recent"] *= pop_scale
        v["test_freq"] *= freq_scale
        variants.append(v)
    return variants


def tune_weights(train: pd.DataFrame, test: pd.DataFrame, name: str, val_rows: int, chunk_rows: int):
    log(f"Tuning weights for {name}")
    hist_train, val, fake_test, true_pos = make_validation(train, test, name, val_rows, seed=20260523)
    history = build_history(hist_train, fake_test)
    src = fake_test["src"].to_numpy(np.int64, copy=False)
    times = fake_test["time"].to_numpy(np.int64, copy=False)
    candidates = fake_test[CAND_COLS].to_numpy(np.int64, copy=False)

    best_score = -1.0
    best_weights = base_weights(name)
    for idx, weights in enumerate(candidate_weight_sets(name)):
        scores_parts = []
        for start in range(0, len(fake_test), chunk_rows):
            end = min(start + chunk_rows, len(fake_test))
            scores_parts.append(score_batch(history, src[start:end], times[start:end], candidates[start:end], weights))
        scores = np.vstack(scores_parts)
        score = mrr_from_scores(scores, true_pos)
        log(f"  {name} weight set {idx}: validation MRR={score:.6f}")
        if score > best_score:
            best_score = score
            best_weights = weights
    log(f"Selected {name} validation MRR={best_score:.6f}")
    return best_weights, best_score


def build_ranker_matrix(history: History, fake_test: pd.DataFrame, true_pos, chunk_rows: int, base_weights):
    src = fake_test["src"].to_numpy(np.int64, copy=False)
    times = fake_test["time"].to_numpy(np.int64, copy=False)
    candidates = fake_test[CAND_COLS].to_numpy(np.int64, copy=False)
    n, m = candidates.shape
    x = np.empty((n * m, len(RANKER_FEATURE_NAMES)), dtype=np.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        x[start * m : end * m] = feature_batch(
            history,
            src[start:end],
            times[start:end],
            candidates[start:end],
            base_score_weights=base_weights,
        )
        log(f"  built ranker features {end}/{n}")
    y = np.zeros(n * m, dtype=np.float32)
    y[np.arange(n) * m + true_pos] = 1.0
    group = np.full(n, m, dtype=np.uint32)
    return x, y, group


def ensure_xgboost():
    try:
        import xgboost as xgb

        return xgb
    except ImportError:
        log("xgboost is missing; installing it in the Kaggle kernel")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
        import xgboost as xgb

        return xgb


def make_xgb_ranker(xgb, args, use_gpu: bool):
    major = int(str(xgb.__version__).split(".")[0])
    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
        "n_estimators": args.ranker_trees,
        "max_depth": args.ranker_depth,
        "learning_rate": args.ranker_lr,
        "subsample": 0.86,
        "colsample_bytree": 0.88,
        "min_child_weight": 28,
        "reg_lambda": 6.0,
        "random_state": args.ranker_seed,
        "n_jobs": 4,
    }
    if use_gpu:
        if major >= 2:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        else:
            params["tree_method"] = "gpu_hist"
            params["predictor"] = "gpu_predictor"
    else:
        params["tree_method"] = "hist"
    return xgb.XGBRanker(**params)


def predict_ranker_scores(model, x, n_rows: int, n_cols: int):
    scores = model.predict(x).astype(np.float32, copy=False).reshape(n_rows, n_cols)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def train_ranker(train: pd.DataFrame, test: pd.DataFrame, name: str, args):
    if name != "dataset2":
        return None
    log(f"Training XGBoost GPU ranker for {name}")
    base_weights = fixed_best_dataset2_weights()
    hist_train, _val, fake_test, true_pos = make_validation(
        train,
        test,
        name,
        args.ranker_rows,
        seed=args.ranker_seed,
        candidate_mode=args.ranker_candidate_mode,
    )
    if len(fake_test) < 1000:
        log("  ranker validation set too small; using fixed V18 weights")
        return {
            "use_ranker": False,
            "reason": "too_few_rows",
            "base_weights": base_weights,
            "validation_mrr": 0.419911417939297,
        }

    split = max(1, min(len(fake_test) - 1, int(len(fake_test) * args.ranker_train_frac)))
    train_fake = fake_test.iloc[:split].reset_index(drop=True)
    eval_fake = fake_test.iloc[split:].reset_index(drop=True)
    train_true = true_pos[:split]
    eval_true = true_pos[split:]

    history = build_history(hist_train, fake_test)
    log(f"  ranker train groups={len(train_fake)}, eval groups={len(eval_fake)}")
    x_train, y_train, group_train = build_ranker_matrix(
        history, train_fake, train_true, args.ranker_chunk_rows, base_weights
    )
    x_eval, y_eval, group_eval = build_ranker_matrix(
        history, eval_fake, eval_true, args.ranker_chunk_rows, base_weights
    )

    base_norm_idx = RANKER_FEATURE_NAMES.index("v18_norm")
    base_eval = x_eval[:, base_norm_idx].reshape(len(eval_fake), 100)
    base_mrr = mrr_from_scores(base_eval, eval_true)
    log(f"  V18 feature baseline MRR on ranker eval={base_mrr:.6f}")

    xgb = ensure_xgboost()
    model = make_xgb_ranker(xgb, args, use_gpu=True)
    try:
        model.fit(
            x_train,
            y_train,
            group=group_train,
            eval_set=[(x_eval, y_eval)],
            eval_group=[group_eval],
            verbose=True,
        )
    except Exception as exc:
        log(f"  GPU ranker fit failed: {exc}; retrying with CPU hist")
        model = make_xgb_ranker(xgb, args, use_gpu=False)
        model.fit(
            x_train,
            y_train,
            group=group_train,
            eval_set=[(x_eval, y_eval)],
            eval_group=[group_eval],
            verbose=True,
        )

    model_scores = predict_ranker_scores(model, x_eval, len(eval_fake), 100)
    model_norm = normalize_rows(model_scores)
    best_alpha = 0.0
    best_mrr = base_mrr
    model_mrr = mrr_from_scores(model_norm, eval_true)
    log(f"  raw ranker MRR on ranker eval={model_mrr:.6f}")
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = alpha * model_norm + (1.0 - alpha) * base_eval
        score = mrr_from_scores(blended, eval_true)
        log(f"  blend alpha={alpha:.2f} MRR={score:.6f}")
        if score > best_mrr:
            best_mrr = score
            best_alpha = float(alpha)

    use_ranker = best_mrr >= base_mrr + args.ranker_min_gain and best_alpha > 0.0
    log(
        f"  selected ranker use={use_ranker} alpha={best_alpha:.2f} "
        f"base={base_mrr:.6f} best={best_mrr:.6f}"
    )
    return {
        "use_ranker": bool(use_ranker),
        "model": model,
        "blend_alpha": best_alpha,
        "validation_mrr": best_mrr,
        "base_validation_mrr": base_mrr,
        "model_validation_mrr": model_mrr,
        "candidate_mode": args.ranker_candidate_mode,
        "train_groups": int(len(train_fake)),
        "eval_groups": int(len(eval_fake)),
        "feature_names": RANKER_FEATURE_NAMES,
        "base_weights": base_weights,
    }


def write_predictions(history, test, weights, output_file: Path, chunk_rows: int):
    src = test["src"].to_numpy(np.int64, copy=False)
    times = test["time"].to_numpy(np.int64, copy=False)
    candidates = test[CAND_COLS].to_numpy(np.int64, copy=False)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="\n") as f:
        for start in range(0, len(test), chunk_rows):
            end = min(start + chunk_rows, len(test))
            scores = score_batch(history, src[start:end], times[start:end], candidates[start:end], weights)
            probs = normalize_rows(scores)
            for row in probs:
                f.write(",".join(f"{x:.8f}" for x in row))
                f.write("\n")
            log(f"  wrote rows {end}/{len(test)} to {output_file.name}")


def write_ranker_predictions(history, test, ranker_info, output_file: Path, chunk_rows: int):
    src = test["src"].to_numpy(np.int64, copy=False)
    times = test["time"].to_numpy(np.int64, copy=False)
    candidates = test[CAND_COLS].to_numpy(np.int64, copy=False)
    base_weights = ranker_info["base_weights"]
    model = ranker_info["model"]
    alpha = float(ranker_info["blend_alpha"])
    base_norm_idx = RANKER_FEATURE_NAMES.index("v18_norm")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="\n") as f:
        for start in range(0, len(test), chunk_rows):
            end = min(start + chunk_rows, len(test))
            x = feature_batch(
                history,
                src[start:end],
                times[start:end],
                candidates[start:end],
                base_score_weights=base_weights,
            )
            model_scores = predict_ranker_scores(model, x, end - start, candidates.shape[1])
            model_norm = normalize_rows(model_scores)
            base_norm = x[:, base_norm_idx].reshape(end - start, candidates.shape[1])
            probs = normalize_rows(alpha * model_norm + (1.0 - alpha) * base_norm)
            for row in probs:
                f.write(",".join(f"{value:.8f}" for value in row))
                f.write("\n")
            log(f"  wrote ranker rows {end}/{len(test)} to {output_file.name}")


def serializable_ranker_info(ranker_info):
    if not ranker_info:
        return None
    return {key: value for key, value in ranker_info.items() if key != "model"}


def solve_dataset(data_dir: Path, output_dir: Path, name: str, args):
    log(f"Reading {name}")
    train, test = read_dataset(data_dir, name)
    ranker_info = None
    scorer = "weighted_heuristic"
    if args.use_ranker and name == "dataset2":
        try:
            ranker_info = train_ranker(train, test, name, args)
        except Exception as exc:
            log(f"Ranker training failed: {exc}; falling back to V18 weights")
            ranker_info = {
                "use_ranker": False,
                "reason": str(exc),
                "base_weights": fixed_best_dataset2_weights(),
                "validation_mrr": 0.419911417939297,
            }

    if ranker_info is not None:
        weights = ranker_info["base_weights"]
        val_score = ranker_info.get("validation_mrr")
        if ranker_info.get("use_ranker"):
            scorer = "xgboost_ranker_blend"
        else:
            log("Using V18 fallback weights for dataset2")
    elif args.tune:
        fixed = fixed_best_weights(name)
        if fixed is None:
            weights, val_score = tune_weights(train, test, name, args.val_rows, args.val_chunk_rows)
        else:
            weights, val_score = fixed
            log(f"Using fixed best weights for {name}: validation MRR={val_score:.6f}")
    else:
        weights, val_score = base_weights(name), None

    log(f"Building full history for {name}")
    history = build_history(train, test)
    out_file = output_dir / f"{name}.csv"
    log(f"Scoring {name}")
    if ranker_info is not None and ranker_info.get("use_ranker"):
        write_ranker_predictions(history, test, ranker_info, out_file, args.ranker_chunk_rows)
    else:
        write_predictions(history, test, weights, out_file, args.chunk_rows)
    return {
        "dataset": name,
        "rows": int(len(test)),
        "scorer": scorer,
        "weights": weights,
        "validation_mrr": val_score,
        "ranker": serializable_ranker_info(ranker_info),
        "output": str(out_file),
    }


def zip_result(output_dir: Path, result_zip: Path):
    if result_zip.exists():
        result_zip.unlink()
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    log(f"Created {result_zip} ({result_zip.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("/tmp/track1_data"))
    parser.add_argument("--zip_path", type=Path, default=Path("/tmp/track1_data.zip"))
    parser.add_argument("--output_dir", type=Path, default=Path("/tmp/track1_outputs"))
    parser.add_argument("--result_zip", type=Path, default=Path("result.zip"))
    parser.add_argument("--chunk_rows", type=int, default=12000)
    parser.add_argument("--val_chunk_rows", type=int, default=8000)
    parser.add_argument("--val_rows", type=int, default=80000)
    parser.add_argument("--no_tune", action="store_true")
    parser.add_argument("--no_ranker", action="store_true")
    parser.add_argument("--ranker_rows", type=int, default=160000)
    parser.add_argument("--ranker_chunk_rows", type=int, default=3000)
    parser.add_argument("--ranker_candidate_mode", choices=["flat", "row", "hybrid"], default="row")
    parser.add_argument("--ranker_train_frac", type=float, default=0.90)
    parser.add_argument("--ranker_seed", type=int, default=20260525)
    parser.add_argument("--ranker_trees", type=int, default=580)
    parser.add_argument("--ranker_depth", type=int, default=5)
    parser.add_argument("--ranker_lr", type=float, default=0.030)
    parser.add_argument("--ranker_min_gain", type=float, default=0.00005)
    args = parser.parse_args()
    args.tune = not args.no_tune
    args.use_ranker = not args.no_ranker

    data_dir = ensure_data(args.data_dir, args.zip_path, DATA_URL)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": []}

    for name in ["dataset1", "dataset2"]:
        metadata["datasets"].append(solve_dataset(data_dir, args.output_dir, name, args))

    zip_result(args.output_dir, args.result_zip)
    metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["result_zip"] = str(args.result_zip)
    with Path("metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log("Done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {exc}")
        raise
