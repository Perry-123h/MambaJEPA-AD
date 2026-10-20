"""Auditable BankSim experiments for the MambaJEPA-AD revision.

This is an independent reconstruction from the manuscript description.  The
selective diagonal state-space recurrence is Mamba-inspired; it is not the
official mamba-ssm package or a fused Mamba kernel.  Every reported number is
written from an actual run of this file.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import platform
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

EXPECTED_SHA256 = "e37006f76d993bfaec3a02d717b4f0bdc1ebfa5d36449e91fb7c07e225278377"
SPLITS = {"train": (0, 107), "validation": (108, 143), "test": (144, 179)}
NUMERIC_NAMES = [
    "log_amount", "normalized_step", "history_count", "history_mean",
    "history_std", "history_max", "recency", "category_repeat",
]


class Measurement:
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.running = True
        self.start = time.perf_counter()

        def sample():
            while self.running:
                self.peak = max(self.peak, self.process.memory_info().rss)
                time.sleep(0.01)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.seconds = time.perf_counter() - self.start
        self.running = False
        self.thread.join()

    def result(self):
        return {"seconds": self.seconds, "peak_process_rss_MiB": self.peak / 2**20}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare(df: pd.DataFrame, history_length: int = 12, category_blind: bool = False):
    """Create strict past-only histories; same-step events never see one another."""
    df = df.copy().reset_index(names="source_row")
    df = df.sort_values(["step", "source_row"], kind="stable").reset_index(drop=True)
    for col in ["customer", "merchant", "age", "gender", "category"]:
        df[col] = df[col].astype(str).str.strip("'")
    step = df.step.to_numpy(np.int16)
    amount = np.log1p(df.amount.to_numpy(np.float64)).astype(np.float32)
    train = step <= 107
    vocab = {
        col: sorted(df.loc[train, col].unique().tolist())
        for col in ["age", "gender", "category"]
    }
    codes = {
        col: df[col].map({v: i + 1 for i, v in enumerate(vocab[col])}).fillna(0).to_numpy(np.int16)
        for col in vocab
    }

    n = len(df)
    history = np.full((n, history_length), -1, dtype=np.int32)
    valid = np.zeros((n, history_length), dtype=bool)
    numeric = np.zeros((n, len(NUMERIC_NAMES)), dtype=np.float32)
    customers = df.customer.to_numpy(dtype=str)
    buffers: dict[str, list[int]] = {}

    for current_step in np.unique(step):
        rows = np.flatnonzero(step == current_step)
        for i in rows:
            prior = buffers.get(customers[i], [])
            k = len(prior)
            numeric[i, 0] = amount[i]
            numeric[i, 1] = current_step / 179.0
            if k:
                h = np.asarray(prior, dtype=np.int32)
                history[i, -k:] = h
                valid[i, -k:] = True
                ha = amount[h]
                numeric[i, 2:] = [
                    k,
                    ha.mean(),
                    ha.std(),
                    ha.max(),
                    min(float(current_step - step[h[-1]]), 179.0) / 179.0,
                    float((codes["category"][h] == codes["category"][i]).mean()),
                ]
            else:
                numeric[i, 6] = 1.0
        for i in rows:
            buf = buffers.setdefault(customers[i], [])
            buf.append(int(i))
            if len(buf) > history_length:
                del buf[0]

    mu = numeric[train].mean(0)
    sd = numeric[train].std(0)
    sd[sd < 1e-8] = 1.0
    scaled = (numeric - mu) / sd

    attr_parts = [scaled]
    field_groups = [{"name": name, "start": i, "end": i + 1, "replacement": "0 after train-standardization"}
                    for i, name in enumerate(NUMERIC_NAMES)]
    cursor = scaled.shape[1]
    for col in ["age", "gender", "category"]:
        if category_blind and col == "category":
            continue
        one_hot = np.eye(len(vocab[col]) + 1, dtype=np.float32)[codes[col]]
        attr_parts.append(one_hot)
        field_groups.append({
            "name": col,
            "start": cursor,
            "end": cursor + one_hot.shape[1],
            "replacement": "all-zero vector, distinct from every valid one-hot level",
        })
        cursor += one_hot.shape[1]
    x = np.concatenate(attr_parts, axis=1).astype(np.float32)
    if category_blind:
        x[:, 7] = 0.0

    safe = history.clip(min=0)
    seq = np.stack([(amount[safe] - mu[0]) / sd[0], np.minimum(step[:, None] - step[safe], 179) / 179.0], axis=-1).astype(np.float32)
    seq[~valid] = 0.0
    hcat = codes["category"][safe].astype(np.int64)
    hcat[~valid] = 0
    if category_blind:
        hcat[:] = 0

    return {
        "x": x,
        "seq": seq,
        "hcat": hcat,
        "valid": valid,
        "y": df.fraud.to_numpy(np.float32),
        "step": step,
        "customer": customers,
        "row_id": df.source_row.to_numpy(np.int32),
        "train": train,
        "validation": (step >= 108) & (step <= 143),
        "test": step >= 144,
        "vocab": vocab,
        "mu": mu,
        "sd": sd,
        "field_groups": field_groups,
        "category_blind": category_blind,
    }


class AttributeEncoder(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 64), nn.GELU(), nn.Linear(64, 32))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class SelectiveSSM(nn.Module):
    """Input-dependent stable diagonal SSM evaluated as a causal recurrence."""
    def __init__(self, input_size=24, state_size=24):
        super().__init__()
        self.in_proj = nn.Linear(input_size, state_size)
        self.dt_proj = nn.Linear(state_size, state_size)
        self.b_proj = nn.Linear(state_size, state_size)
        self.c_proj = nn.Linear(state_size, state_size)
        self.log_a = nn.Parameter(torch.zeros(state_size))
        self.skip = nn.Parameter(torch.ones(state_size))

    def forward(self, events, valid):
        h = events.new_zeros((events.shape[0], self.log_a.numel()))
        out = h
        a = -torch.exp(self.log_a).unsqueeze(0)
        for j in range(events.shape[1]):
            u = F.silu(self.in_proj(events[:, j]))
            dt = F.softplus(self.dt_proj(u))
            decay = torch.exp(dt * a)
            b_t = torch.sigmoid(self.b_proj(u))
            c_t = torch.sigmoid(self.c_proj(u))
            proposal = decay * h + (1.0 - decay) * b_t * u
            y = c_t * proposal + self.skip * u
            keep = valid[:, j, None]
            h = torch.where(keep, proposal, h)
            out = torch.where(keep, y, out)
        return out


class DualViewModel(nn.Module):
    def __init__(self, features: int, category_count: int, variant="full", sequence_kind="ssm"):
        super().__init__()
        self.variant = variant
        self.sequence_kind = sequence_kind
        self.attribute = AttributeEncoder(features)
        self.target = copy.deepcopy(self.attribute)
        for p in self.target.parameters():
            p.requires_grad = False
        self.category_embedding = nn.Embedding(category_count + 1, 6, padding_idx=0)
        self.event_projection = nn.Linear(8, 24)
        if sequence_kind == "ssm":
            self.sequence = SelectiveSSM(24, 24)
        elif sequence_kind == "lstm":
            self.sequence = nn.LSTM(24, 24, batch_first=True)
        elif sequence_kind == "transformer":
            layer = nn.TransformerEncoderLayer(24, 4, 48, dropout=0.0, activation="gelu", batch_first=True)
            self.sequence = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.position = nn.Parameter(torch.zeros(1, 12, 24))
            nn.init.normal_(self.position, std=0.02)
        elif sequence_kind == "none":
            self.sequence = None
        else:
            raise ValueError(sequence_kind)
        self.sequence_projection = nn.Linear(24, 32)
        self.attribute_predictor = nn.Linear(32, 32)
        self.sequence_predictor = nn.Linear(32, 32)
        self.gate = nn.Linear(64, 32)
        self.risk = nn.Sequential(nn.Linear(66, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, x, seq, hcat, valid):
        a = self.attribute(x)
        if self.sequence_kind == "none" or self.variant == "no_behavior":
            return a, torch.zeros_like(a)
        events = F.gelu(self.event_projection(torch.cat([seq, self.category_embedding(hcat)], dim=-1)))
        if self.sequence_kind == "ssm":
            raw = self.sequence(events, valid)
        elif self.sequence_kind == "lstm":
            lengths = valid.sum(1)
            compact = torch.zeros_like(events)
            for k in range(1, events.shape[1] + 1):
                selected = lengths == k
                compact[selected, :k] = events[selected, -k:]
            packed = nn.utils.rnn.pack_padded_sequence(compact, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.sequence(packed)
            raw = hidden[-1] * (lengths > 0).unsqueeze(1)
        else:
            safe_valid = valid.clone()
            safe_valid[:, 0] |= ~valid.any(1)
            causal = torch.triu(torch.ones(events.shape[1], events.shape[1], dtype=torch.bool, device=events.device), diagonal=1)
            encoded = self.sequence(events + self.position[:, :events.shape[1]], mask=causal, src_key_padding_mask=~safe_valid)
            raw = encoded[:, -1] * valid.any(1).unsqueeze(1)
        return a, F.normalize(self.sequence_projection(raw), dim=-1)

    def forward(self, x, seq, hcat, valid):
        a, b = self.encode(x, seq, hcat, valid)
        gate = torch.sigmoid(self.gate(torch.cat([a, b], 1)))
        fused = (a + b) / 2 if self.variant == "no_gate" else gate * a + (1 - gate) * b
        if self.variant == "no_behavior" or self.sequence_kind == "none":
            fused = a
        disagreement = (1 - (a * b).sum(1, keepdim=True))
        distance = torch.linalg.vector_norm(a - b, dim=1, keepdim=True)
        return self.risk(torch.cat([fused, a, disagreement, distance], 1)).squeeze(1)


def as_tensors(data):
    return [torch.from_numpy(data[name]) for name in ["x", "seq", "hcat", "valid"]]


def batch(tensors_, indices):
    return [t[indices] for t in tensors_]


def infer(model, tensors_, indices, batch_size):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            values.append(torch.sigmoid(model(*batch(tensors_, indices[start:start + batch_size]))).cpu().numpy())
    return np.concatenate(values)


def representation_diagnostics(model, tensors_, indices):
    model.eval()
    with torch.no_grad():
        a, b = model.encode(*batch(tensors_, indices[:8192]))

    def stats(z):
        z = z.cpu().numpy().astype(np.float64)
        centered = z - z.mean(0)
        singular = np.linalg.svd(centered, compute_uv=False)
        probability = singular / max(singular.sum(), 1e-15)
        n = len(z)
        pair_cos = ((z.sum(0) ** 2).sum() - (z * z).sum()) / max(n * (n - 1), 1)
        return {
            "mean_dimension_std": float(z.std(0).mean()),
            "effective_rank": float(np.exp(-(probability * np.log(probability + 1e-15)).sum())),
            "mean_pair_cosine": float(pair_cos),
        }
    return {"attribute": stats(a), "sequence": stats(b)}


def save_predictions(folder, data, split, scores):
    folder.mkdir(parents=True, exist_ok=True)
    selected = data[split]
    frame = pd.DataFrame({
        "row_id": data["row_id"][selected],
        "step": data["step"][selected],
        "customer": data["customer"][selected],
        "label": data["y"][selected].astype(np.int8),
        "score": scores,
    })
    frame.to_csv(folder / f"{split}_predictions.csv.gz", index=False, compression="gzip")


def choose_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[int(np.argmax(f1))])


def metrics(yv, pv, yt, pt):
    threshold = choose_threshold(yv, pv)
    predicted = pt >= threshold
    tn, fp, fn, tp = confusion_matrix(yt, predicted, labels=[0, 1]).ravel()
    return {
        "validation_AUPRC": float(average_precision_score(yv, pv)),
        "validation_selected_F1_threshold": threshold,
        "test_ROC_AUC": float(roc_auc_score(yt, pt)),
        "test_AUPRC": float(average_precision_score(yt, pt)),
        "test_F1": float(f1_score(yt, predicted)),
        "test_MCC": float(matthews_corrcoef(yt, predicted)),
        "test_FPR": float(fp / max(fp + tn, 1)),
        "test_precision": float(tp / max(tp + fp, 1)),
        "test_recall": float(tp / max(tp + fn, 1)),
    }


def mask_fields(x, groups, ratio, generator):
    """Mask semantic fields, never individual one-hot coordinates."""
    masked = x.clone()
    decisions = torch.rand((len(x), len(groups)), generator=generator) < ratio
    for field, group in enumerate(groups):
        masked[:, group["start"]:group["end"]] *= (~decisions[:, field, None])
    return masked, decisions


def train_neural(data, cfg, out, seed, variant="full", sequence_kind="ssm", pretrain=True):
    seed_all(seed)
    folder = out
    folder.mkdir(parents=True, exist_ok=True)
    tensors_ = as_tensors(data)
    tr = np.flatnonzero(data["train"])
    va = np.flatnonzero(data["validation"])
    te = np.flatnonzero(data["test"])
    y_tensor = torch.from_numpy(data["y"])
    model = DualViewModel(data["x"].shape[1], len(data["vocab"]["category"]), variant, sequence_kind)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 10000)
    log = {
        "seed": seed,
        "variant": variant,
        "sequence_kind": sequence_kind,
        "pretraining_enabled": bool(pretrain),
        "initial_representation_diagnostics": representation_diagnostics(model, tensors_, tr),
        "field_mask_definition": data["field_groups"],
    }
    pretrain_log = []
    pretrain_measurement = {"seconds": 0.0, "peak_process_rss_MiB": psutil.Process().memory_info().rss / 2**20}
    if pretrain:
        with Measurement() as measured:
            for epoch in range(cfg["pretrain_epochs"]):
                model.train()
                losses = []
                masked_fractions = []
                for order in torch.randperm(len(tr), generator=generator).split(cfg["batch_size"]):
                    indices = tr[order.numpy()]
                    x, seq, hcat, valid = batch(tensors_, indices)
                    masked, decisions = mask_fields(x, data["field_groups"], cfg["mask_ratio"], generator)
                    a, b = model.encode(masked, seq, hcat, valid)
                    with torch.no_grad():
                        target = model.target(x)
                    loss_attribute = (1 - F.cosine_similarity(model.attribute_predictor(a), target)).mean()
                    loss = loss_attribute
                    loss_sequence = torch.tensor(0.0)
                    loss_alignment = torch.tensor(0.0)
                    if variant != "no_behavior" and sequence_kind != "none":
                        loss_sequence = (1 - F.cosine_similarity(model.sequence_predictor(b), target)).mean()
                        loss = loss + 0.5 * loss_sequence
                        if variant != "no_alignment":
                            loss_alignment = (1 - F.cosine_similarity(a, b)).mean()
                            loss = loss + 0.2 * loss_alignment
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    optimizer.step()
                    with torch.no_grad():
                        for target_param, online_param in zip(model.target.parameters(), model.attribute.parameters()):
                            target_param.mul_(cfg["ema_decay"]).add_(online_param, alpha=1 - cfg["ema_decay"])
                    losses.append([float(loss), float(loss_attribute), float(loss_sequence), float(loss_alignment)])
                    masked_fractions.append(float(decisions.float().mean()))
                mean_losses = np.mean(losses, axis=0)
                pretrain_log.append({
                    "epoch": epoch + 1,
                    "loss": float(mean_losses[0]),
                    "attribute_prediction_loss": float(mean_losses[1]),
                    "sequence_prediction_loss": float(mean_losses[2]),
                    "alignment_loss": float(mean_losses[3]),
                    "observed_mask_fraction": float(np.mean(masked_fractions)),
                })
        pretrain_measurement = measured.result()
        log["post_pretraining_representation_diagnostics"] = representation_diagnostics(model, tensors_, tr)

    class_weight = float(np.sqrt((data["y"][tr] == 0).sum() / (data["y"][tr] == 1).sum()))
    best_ap = -1.0
    best_epoch = 0
    best_state = None
    patience = 0
    supervised_log = []
    with Measurement() as measured:
        for epoch in range(cfg["finetune_epochs"]):
            model.train()
            losses = []
            for order in torch.randperm(len(tr), generator=generator).split(cfg["batch_size"]):
                indices = tr[order.numpy()]
                logits = model(*batch(tensors_, indices))
                loss = F.binary_cross_entropy_with_logits(logits, y_tensor[indices], pos_weight=torch.tensor(class_weight))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                optimizer.step()
                losses.append(float(loss.detach()))
            pv = infer(model, tensors_, va, cfg["inference_batch_size"])
            ap = float(average_precision_score(data["y"][va], pv))
            supervised_log.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses)), "validation_AUPRC": ap})
            if ap > best_ap + 1e-6:
                best_ap = ap
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= cfg["early_stopping_patience"]:
                    break
    model.load_state_dict(best_state)
    pv = infer(model, tensors_, va, cfg["inference_batch_size"])
    pt = infer(model, tensors_, te, cfg["inference_batch_size"])
    infer(model, tensors_, te, cfg["inference_batch_size"])
    repeats = []
    with Measurement() as inference_measure:
        for _ in range(5):
            start = time.perf_counter()
            infer(model, tensors_, te, cfg["inference_batch_size"])
            repeats.append(time.perf_counter() - start)
    torch.save({"state_dict": model.state_dict(), "seed": seed, "variant": variant, "sequence_kind": sequence_kind,
                "feature_dimension": data["x"].shape[1]}, folder / "model.pt")
    save_predictions(folder, data, "validation", pv)
    save_predictions(folder, data, "test", pt)
    log.update({
        "pretraining": pretrain_measurement,
        "pretraining_epochs": pretrain_log,
        "supervised_training": measured.result(),
        "supervised_epochs": supervised_log,
        "best_epoch": best_epoch,
        "positive_class_weight": class_weight,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "final_representation_diagnostics": representation_diagnostics(model, tensors_, tr),
        "inference": {"five_test_pass_seconds": repeats, "median_seconds": float(np.median(repeats)),
                      "peak_process_rss_MiB": inference_measure.peak / 2**20, "test_transactions": len(te)},
        **metrics(data["y"][va], pv, data["y"][te], pt),
    })
    (folder / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return pv, pt, log


def train_tree(data, out, name, config):
    folder = out / name.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    tr, va, te = data["train"], data["validation"], data["test"]
    constructors = {
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier,
        "CatBoost": CatBoostClassifier,
        "Random Forest": RandomForestClassifier,
    }
    model = constructors[name](**config)
    with Measurement() as training:
        model.fit(data["x"][tr], data["y"][tr])

    def predict(array):
        return np.concatenate([model.predict_proba(array[i:i + 32768])[:, 1] for i in range(0, len(array), 32768)])

    pv, pt = predict(data["x"][va]), predict(data["x"][te])
    predict(data["x"][te])
    repeats = []
    with Measurement() as inference:
        for _ in range(5):
            start = time.perf_counter()
            predict(data["x"][te])
            repeats.append(time.perf_counter() - start)
    with open(folder / "model.pkl", "wb") as handle:
        pickle.dump(model, handle)
    save_predictions(folder, data, "validation", pv)
    save_predictions(folder, data, "test", pt)
    report = {
        "configuration": config,
        "tuning_budget": "one prespecified configuration; no hyperparameter search",
        "training": training.result(),
        "inference": {"five_test_pass_seconds": repeats, "median_seconds": float(np.median(repeats)),
                      "peak_process_rss_MiB": inference.peak / 2**20, "test_transactions": int(te.sum())},
        **metrics(data["y"][va], pv, data["y"][te], pt),
    }
    (folder / "training_log.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return pv, pt, report


def fuse(data, out, tree_scores, neural_scores, label, step):
    xv, xt = tree_scores
    nv, nt = neural_scores
    grid = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    yv, yt = data["y"][data["validation"]], data["y"][data["test"]]
    validation_ap = [float(average_precision_score(yv, alpha * xv + (1 - alpha) * nv)) for alpha in grid]
    alpha = float(grid[int(np.argmax(validation_ap))])
    pv, pt = alpha * xv + (1 - alpha) * nv, alpha * xt + (1 - alpha) * nt
    folder = out / label.replace(" ", "_")
    save_predictions(folder, data, "validation", pv)
    save_predictions(folder, data, "test", pt)
    report = {
        "formula": "score = alpha * XGBoost + (1-alpha) * neural",
        "alpha_grid": grid.tolist(),
        "validation_AUPRC_grid": validation_ap,
        "validation_selected_alpha": alpha,
        "fusion_compute_seconds": "measured by src/postprocess_results.py for the supplied run",
        **metrics(yv, pv, yt, pt),
    }
    (folder / "fusion_log.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return pv, pt, report


def clustered_bootstrap(data, predictions, resamples, seed):
    selected = data["test"]
    y = data["y"][selected]
    _, customer_code = np.unique(data["customer"][selected], return_inverse=True)
    groups = int(customer_code.max() + 1)
    rng = np.random.default_rng(seed)
    names = list(predictions)
    values = {name: [] for name in names}
    for _ in range(resamples):
        counts = np.bincount(rng.integers(groups, size=groups), minlength=groups)
        weights = counts[customer_code]
        for name in names:
            values[name].append(average_precision_score(y, predictions[name], sample_weight=weights))
    result = {"resamples": resamples, "seed": seed, "customer_clusters": groups, "AUPRC_95_CI": {}}
    for name in names:
        result["AUPRC_95_CI"][name] = np.quantile(values[name], [0.025, 0.975]).tolist()
    base = np.asarray(values["XGBoost"])
    result["paired_difference_vs_XGBoost_95_CI"] = {
        name: np.quantile(np.asarray(values[name]) - base, [0.025, 0.975]).tolist()
        for name in names if name != "XGBoost"
    }
    return result


def summarize_seed_runs(runs):
    values = np.asarray([run["test_AUPRC"] for run in runs], dtype=float)
    return {"seeds": [run["seed"] for run in runs], "test_AUPRC_values": values.tolist(),
            "mean_test_AUPRC": float(values.mean()), "sample_SD_test_AUPRC": float(values.std(ddof=1)) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/revision.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    os.environ["OMP_NUM_THREADS"] = str(cfg["threads"])
    os.environ["MKL_NUM_THREADS"] = str(cfg["threads"])
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.data)
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha256 != EXPECTED_SHA256:
        raise SystemExit(f"Dataset checksum mismatch: {sha256}")

    with Measurement() as csv_measure:
        df = pd.read_csv(source)
    if len(df) != 594643 or int(df.fraud.sum()) != 7200:
        raise SystemExit("Unexpected BankSim row or fraud count")
    with Measurement() as preprocessing_measure:
        data = prepare(df, cfg["history_length"], category_blind=False)
    del df
    metadata = {
        "status": "independent reconstruction from the paper; not original author training code",
        "dataset_source": "https://www.kaggle.com/datasets/ealaxi/banksim1",
        "dataset_file": source.name,
        "dataset_SHA256": sha256,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "hardware": {"processor": platform.processor(), "logical_CPUs": os.cpu_count(),
                     "RAM_GiB": psutil.virtual_memory().total / 2**30, "threads_used": cfg["threads"], "platform": platform.platform()},
        "config": cfg,
        "data_loading": csv_measure.result(),
        "history_and_feature_construction": preprocessing_measure.result(),
        "split": {name: {"steps": [lo, hi], "transactions": int(data[name].sum()),
                          "frauds": int(data["y"][data[name]].sum())} for name, (lo, hi) in SPLITS.items()},
        "feature_dimension": int(data["x"].shape[1]),
        "categorical_vocabularies": data["vocab"],
        "masking": {"unit": "semantic field group", "probability_per_field": cfg["mask_ratio"],
                    "independent_across_examples_and_fields": True, "groups": data["field_groups"]},
        "models": {},
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    train_y = data["y"][data["train"]]
    weight = float(np.sqrt((train_y == 0).sum() / (train_y == 1).sum()))
    tree_configs = {
        "XGBoost": dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, reg_alpha=0.0,
                        scale_pos_weight=weight, tree_method="hist", n_jobs=cfg["threads"], random_state=42,
                        eval_metric="logloss"),
        "LightGBM": dict(n_estimators=400, num_leaves=31, max_depth=-1, learning_rate=0.05,
                         subsample=1.0, colsample_bytree=1.0, min_child_samples=20, reg_lambda=0.0,
                         scale_pos_weight=weight, n_jobs=cfg["threads"], random_state=42, verbosity=-1),
        "CatBoost": dict(iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
                         loss_function="Logloss", class_weights=[1.0, weight], thread_count=cfg["threads"],
                         random_seed=42, verbose=False, allow_writing_files=False),
        "Random Forest": dict(n_estimators=300, max_depth=16, min_samples_leaf=2, max_features="sqrt",
                              class_weight="balanced", n_jobs=cfg["threads"], random_state=42),
    }
    predictions = {}
    for name, config in tree_configs.items():
        print("TREE", name, flush=True)
        pv, pt, report = train_tree(data, out, name, config)
        metadata["models"][name] = report
        predictions[name] = (pv, pt)
        (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    baseline_specs = [
        ("Balanced MLP", "none", "no_behavior"),
        ("Supervised LSTM", "lstm", "full"),
        ("Supervised Transformer", "transformer", "full"),
        ("Supervised SSM", "ssm", "full"),
    ]
    for name, sequence_kind, variant in baseline_specs:
        print("NEURAL BASELINE", name, flush=True)
        pv, pt, report = train_neural(data, cfg, out / name.replace(" ", "_") / "seed_42", 42,
                                      variant=variant, sequence_kind=sequence_kind, pretrain=False)
        report["tuning_budget"] = "one prespecified configuration; validation checkpoint among at most 8 epochs"
        metadata["models"][name] = report
        predictions[name] = (pv, pt)
        (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    ablation_specs = [
        ("MambaJEPA full", "full", True),
        ("MambaJEPA without SSL", "full", False),
        ("MambaJEPA without behavior", "no_behavior", True),
        ("MambaJEPA without gate", "no_gate", True),
        ("MambaJEPA without alignment", "no_alignment", True),
    ]
    ablation_reports = {}
    ablation_predictions = {}
    for label, variant, pretrain in ablation_specs:
        runs = []
        for seed in cfg["seeds"]:
            print("ABLATION", label, "SEED", seed, flush=True)
            pv, pt, report = train_neural(data, cfg, out / "ablations" / label.replace(" ", "_") / f"seed_{seed}",
                                          seed, variant=variant, sequence_kind="ssm", pretrain=pretrain)
            runs.append(report)
            if seed == cfg["seeds"][0]:
                ablation_predictions[label] = (pv, pt)
        ablation_reports[label] = {"summary": summarize_seed_runs(runs), "runs": runs}
        (out / "ablation_results.json").write_text(json.dumps(ablation_reports, indent=2), encoding="utf-8")
    metadata["ablations"] = ablation_reports
    predictions["MambaJEPA neural"] = ablation_predictions["MambaJEPA full"]

    _, fusion_main_test, fusion_main = fuse(data, out, predictions["XGBoost"], predictions["MambaJEPA neural"],
                                            "XGBoost + MambaJEPA", cfg["fusion_grid_step"])
    _, fusion_control_test, fusion_control = fuse(data, out, predictions["XGBoost"], predictions["Supervised SSM"],
                                                  "XGBoost + supervised SSM", cfg["fusion_grid_step"])
    metadata["models"]["XGBoost + MambaJEPA"] = fusion_main
    metadata["models"]["XGBoost + supervised SSM"] = fusion_control

    bootstrap = clustered_bootstrap(
        data,
        {"XGBoost": predictions["XGBoost"][1], "XGBoost + MambaJEPA": fusion_main_test,
         "XGBoost + supervised SSM": fusion_control_test},
        cfg["bootstrap_resamples"], cfg["bootstrap_seed"],
    )
    metadata["customer_clustered_bootstrap"] = bootstrap

    print("CATEGORY BLIND", flush=True)
    source_df = pd.read_csv(source)
    with Measurement() as blind_prep:
        blind = prepare(source_df, cfg["history_length"], category_blind=True)
    del source_df
    blind_weight = float(np.sqrt((blind["y"][blind["train"]] == 0).sum() / (blind["y"][blind["train"]] == 1).sum()))
    blind_xgb_cfg = dict(tree_configs["XGBoost"])
    blind_xgb_cfg["scale_pos_weight"] = blind_weight
    bxv, bxt, bx_report = train_tree(blind, out / "category_blind", "XGBoost", blind_xgb_cfg)
    bnv, bnt, bn_report = train_neural(blind, cfg, out / "category_blind" / "MambaJEPA_seed_42", 42,
                                       variant="full", sequence_kind="ssm", pretrain=True)
    _, bft, bf_report = fuse(blind, out / "category_blind", (bxv, bxt), (bnv, bnt),
                             "XGBoost + MambaJEPA", cfg["fusion_grid_step"])
    metadata["category_blind"] = {"history_and_feature_construction": blind_prep.result(), "XGBoost": bx_report,
                                  "MambaJEPA neural seed 42": bn_report, "XGBoost + MambaJEPA": bf_report}
    metadata["customer_clustered_bootstrap_category_blind"] = clustered_bootstrap(
        blind, {"XGBoost": bxt, "XGBoost + MambaJEPA": bft}, cfg["bootstrap_resamples"], cfg["bootstrap_seed"]
    )

    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
