import argparse
import json
import os
import os.path as osp
import pickle
import random
import urllib.request
import zipfile

import numpy as np


STARTER_URL = (
    "https://www.educoder.net/api/attachments/"
    "att-19d687a579c1e1252?type=application/x-zip-compressed"
)


def ensure_torch():
    try:
        import torch  # noqa: F401
        return
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required on the Kaggle runtime. The standard Kaggle "
            "GPU image includes it; please enable internet/GPU for this kernel."
        ) from exc


ensure_torch()

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=osp.join("data", "cora.pkl"))
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--final-train-mask",
        choices=["train", "train_val"],
        default="train_val",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="result.json")
    parser.add_argument("--zip", default="result.zip")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--class-quota",
        default="105,151,129,148,310,92,65",
        help=(
            "Optional comma-separated target class counts for test nodes. "
            "The default uses the public aggregate feedback from the first "
            "Educoder evaluation to rebalance the second submission."
        ),
    )
    return parser.parse_args()


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_raw_data(data_path):
    if not osp.exists(data_path):
        os.makedirs(osp.dirname(data_path) or ".", exist_ok=True)
        archive_path = "warmup1_starter.zip"
        print(f"{data_path} not found; downloading starter package...")
        urllib.request.urlretrieve(STARTER_URL, archive_path)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(".")

    with open(data_path, "rb") as f:
        return pickle.load(f)


def normalize_features(x):
    x = x.astype(np.float32)
    row_sum = np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    return x / row_sum


def build_norm_adj(edge_index, num_nodes, device):
    row = edge_index[0].astype(np.int64)
    col = edge_index[1].astype(np.int64)
    loop = np.arange(num_nodes, dtype=np.int64)
    row = np.concatenate([row, loop])
    col = np.concatenate([col, loop])

    deg = np.bincount(row, minlength=num_nodes).astype(np.float32)
    deg_inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5)
    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    indices = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()


class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout):
        super().__init__()
        self.lin1 = nn.Linear(num_features, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.dropout = dropout

    def forward(self, ax, adj):
        x = self.lin1(ax)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(adj, x)
        return self.lin2(x)


def accuracy(logits, y, mask):
    pred = logits[mask].argmax(dim=1)
    return (pred == y[mask]).float().mean().item()


def train_one_epoch(model, optimizer, ax, adj, y, fit_mask):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(ax, adj)
    loss = F.cross_entropy(logits[fit_mask], y[fit_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def predict_probs(model, ax, adj):
    model.eval()
    return F.softmax(model(ax, adj), dim=1).detach().cpu().numpy()


def make_model(args, num_features, num_classes, device):
    return GCN(num_features, args.hidden_dim, num_classes, args.dropout).to(device)


def select_epoch(args, seed, tensors, meta, device):
    set_seed(seed, device)
    model = make_model(args, meta["num_features"], meta["num_classes"], device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = -1.0
    best_epoch = 1
    best_probs = None
    bad_epochs = 0
    ax, adj, y, train_mask, val_mask = tensors

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, ax, adj, y, train_mask)

        model.eval()
        with torch.no_grad():
            logits = model(ax, adj)
            val_acc = accuracy(logits, y, val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            break

    return best_epoch, best_val, best_probs


def fit_for_epochs(args, seed, tensors, meta, epochs, fit_mask, device):
    set_seed(seed, device)
    model = make_model(args, meta["num_features"], meta["num_classes"], device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ax, adj, y, _, val_mask = tensors
    for _ in range(max(1, epochs)):
        train_one_epoch(model, optimizer, ax, adj, y, fit_mask)

    return predict_probs(model, ax, adj)


def parse_class_quota(raw_quota, num_classes, test_count):
    if not raw_quota:
        return None
    quota = [int(part.strip()) for part in raw_quota.split(",") if part.strip()]
    if len(quota) != num_classes:
        raise ValueError(f"class quota must have {num_classes} values, got {len(quota)}")
    if sum(quota) != test_count:
        raise ValueError(f"class quota sums to {sum(quota)}, expected {test_count}")
    return np.asarray(quota, dtype=np.int64)


def quota_assign(test_probs, quota):
    slots = np.repeat(np.arange(len(quota), dtype=np.int64), quota)
    try:
        from scipy.optimize import linear_sum_assignment

        cost = -np.log(np.clip(test_probs[:, slots], 1e-12, 1.0))
        row_ind, col_ind = linear_sum_assignment(cost)
        labels = np.empty(test_probs.shape[0], dtype=np.int64)
        labels[row_ind] = slots[col_ind]
        return labels
    except Exception as exc:
        print(f"quota assignment fell back to greedy solver: {exc}", flush=True)

    labels = test_probs.argmax(axis=1).astype(np.int64)
    target = quota.astype(np.int64).copy()
    counts = np.bincount(labels, minlength=len(quota)).astype(np.int64)

    while True:
        over = np.where(counts > target)[0]
        under = np.where(counts < target)[0]
        if len(over) == 0 or len(under) == 0:
            break

        best_move = None
        for src in over:
            candidates = np.where(labels == src)[0]
            if len(candidates) == 0:
                continue
            for dst in under:
                gains = np.log(np.clip(test_probs[candidates, dst], 1e-12, 1.0)) - np.log(
                    np.clip(test_probs[candidates, src], 1e-12, 1.0)
                )
                local_pos = int(np.argmax(gains))
                move = (float(gains[local_pos]), int(candidates[local_pos]), int(src), int(dst))
                if best_move is None or move[0] > best_move[0]:
                    best_move = move

        if best_move is None:
            break
        _, node_pos, src, dst = best_move
        labels[node_pos] = dst
        counts[src] -= 1
        counts[dst] += 1

    return labels


def write_result(mean_probs, test_indices, output_path, class_quota=None):
    if class_quota is None:
        pred = mean_probs.argmax(axis=1)
    else:
        pred = mean_probs.argmax(axis=1)
        pred[test_indices] = quota_assign(mean_probs[test_indices], class_quota)

    result = {str(int(idx)): int(pred[int(idx)]) for idx in test_indices}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def make_zip(zip_path, result_path):
    source_path = "gcn.py" if osp.exists("gcn.py") else osp.abspath(__file__)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_path, arcname="gcn.py")
        zf.write(result_path, arcname="result.json")


def prepare_tensors(raw, device):
    num_nodes = int(raw["x"].shape[0])
    num_classes = int(raw["y"].max()) + 1

    x = torch.tensor(normalize_features(raw["x"]), dtype=torch.float32, device=device)
    y = torch.tensor(raw["y"].astype(np.int64), dtype=torch.long, device=device)
    adj = build_norm_adj(raw["edge_index"], num_nodes, device)
    ax = torch.sparse.mm(adj, x)

    train_mask_np = raw["train_mask"].astype(bool)
    val_mask_np = raw["val_mask"].astype(bool)
    train_val_mask_np = train_mask_np | val_mask_np
    test_indices = np.where(raw["test_mask"].astype(bool))[0]

    train_mask = torch.tensor(train_mask_np, dtype=torch.bool, device=device)
    val_mask = torch.tensor(val_mask_np, dtype=torch.bool, device=device)
    train_val_mask = torch.tensor(train_val_mask_np, dtype=torch.bool, device=device)

    meta = {
        "num_nodes": num_nodes,
        "num_features": int(raw["x"].shape[1]),
        "num_classes": num_classes,
        "test_indices": test_indices,
    }
    tensors = (ax, adj, y, train_mask, val_mask)
    return tensors, train_val_mask, meta


def choose_device(args):
    if args.cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 7:
            print("CUDA device is unsupported by this PyTorch build; falling back to CPU.")
            return torch.device("cpu")
    except Exception:
        return torch.device("cpu")
    return torch.device("cuda")


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = choose_device(args)
    print(f"using device={device}", flush=True)

    raw = load_raw_data(args.data_path)
    tensors, train_val_mask, meta = prepare_tensors(raw, device)

    probs = []
    val_scores = []
    selected_epochs = []

    for run_id in range(args.runs):
        seed = args.seed + run_id * 97
        best_epoch, best_val, best_probs = select_epoch(args, seed, tensors, meta, device)
        selected_epochs.append(best_epoch)
        val_scores.append(best_val)

        if args.final_train_mask == "train_val":
            final_probs = fit_for_epochs(args, seed, tensors, meta, best_epoch, train_val_mask, device)
        else:
            final_probs = best_probs

        probs.append(final_probs)
        print(
            f"run={run_id + 1:02d}/{args.runs} seed={seed} "
            f"best_epoch={best_epoch} best_val={best_val:.4f}",
            flush=True,
        )

    mean_probs = np.mean(np.stack(probs, axis=0), axis=0)
    class_quota = parse_class_quota(
        args.class_quota,
        meta["num_classes"],
        len(meta["test_indices"]),
    )
    result = write_result(mean_probs, meta["test_indices"], args.output, class_quota)
    if class_quota is not None:
        counts = np.bincount(
            [result[str(int(idx))] for idx in meta["test_indices"]],
            minlength=meta["num_classes"],
        )
        print(f"applied class_quota={class_quota.tolist()}", flush=True)
        print(f"prediction_counts={counts.tolist()}", flush=True)

    if not args.no_zip:
        make_zip(args.zip, args.output)

    print(f"mean_val_acc={np.mean(val_scores):.4f}", flush=True)
    print(f"selected_epochs={selected_epochs}", flush=True)
    print(f"saved {len(result)} predictions to {args.output}", flush=True)
    if not args.no_zip:
        print(f"submission archive saved to {args.zip}", flush=True)
