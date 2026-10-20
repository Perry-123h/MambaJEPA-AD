"""Create reviewer-facing summaries from a completed experiment run."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def load_scores(folder, split):
    frame = pd.read_csv(folder / f"{split}_predictions.csv.gz")
    return frame, frame.score.to_numpy()


def paired_cluster_bootstrap(index, scores, resamples=2000, seed=20260808):
    y = index.label.to_numpy()
    _, customer_code = np.unique(index.customer.astype(str), return_inverse=True)
    group_count = int(customer_code.max() + 1)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in scores}
    for _ in range(resamples):
        counts = np.bincount(rng.integers(group_count, size=group_count), minlength=group_count)
        weights = counts[customer_code]
        for name, score in scores.items():
            values[name].append(average_precision_score(y, score, sample_weight=weights))
    pairs = {}
    names = list(scores)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            difference = np.asarray(values[left]) - np.asarray(values[right])
            pairs[f"{left} minus {right}"] = {
                "point_difference": float(average_precision_score(y, scores[left]) - average_precision_score(y, scores[right])),
                "95_CI": np.quantile(difference, [0.025, 0.975]).tolist(),
            }
    return {"resamples": resamples, "seed": seed, "customer_clusters": group_count, "paired_differences": pairs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = Path(args.results)
    meta_path = root / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model_rows = []
    for name, report in meta["models"].items():
        model_rows.append({
            "model": name,
            "validation_AUPRC": report.get("validation_AUPRC"),
            "validation_selected_F1_threshold": report.get("validation_selected_F1_threshold"),
            "validation_selected_fusion_alpha": report.get("validation_selected_alpha"),
            "test_ROC_AUC": report.get("test_ROC_AUC"),
            "test_AUPRC": report.get("test_AUPRC"),
            "test_F1": report.get("test_F1"),
            "test_MCC": report.get("test_MCC"),
            "test_FPR": report.get("test_FPR"),
            "test_precision": report.get("test_precision"),
            "test_recall": report.get("test_recall"),
        })
    pd.DataFrame(model_rows).to_csv(root / "metrics.csv", index=False)

    ablation_rows = []
    for name, report in meta["ablations"].items():
        summary = report["summary"]
        ablation_rows.append({
            "configuration": name,
            "seeds": ",".join(map(str, summary["seeds"])),
            "test_AUPRC_values": ",".join(f"{x:.9f}" for x in summary["test_AUPRC_values"]),
            "mean_test_AUPRC": summary["mean_test_AUPRC"],
            "sample_SD_test_AUPRC": summary["sample_SD_test_AUPRC"],
        })
    pd.DataFrame(ablation_rows).to_csv(root / "ablation_summary.csv", index=False)

    resources = [
        {"component": "CSV loading", **meta["data_loading"]},
        {"component": "Past-only history and feature construction", **meta["history_and_feature_construction"]},
    ]
    for name, report in meta["models"].items():
        if "training" in report:
            resources.append({"component": f"{name} training", **report["training"]})
        if "pretraining" in report:
            resources.append({"component": f"{name} pretraining", **report["pretraining"]})
        if "supervised_training" in report:
            resources.append({"component": f"{name} supervised training", **report["supervised_training"]})
        if isinstance(report.get("inference"), dict):
            resources.append({"component": f"{name} inference over 133231 test transactions",
                              "seconds": report["inference"].get("median_seconds"),
                              "peak_process_rss_MiB": report["inference"].get("peak_process_rss_MiB")})

    full_seed42 = meta["ablations"]["MambaJEPA full"]["runs"][0]
    resources.append({"component": "MambaJEPA neural seed 42 pretraining", **full_seed42["pretraining"]})
    resources.append({"component": "MambaJEPA neural seed 42 supervised training", **full_seed42["supervised_training"]})
    resources.append({"component": "MambaJEPA neural seed 42 inference over 133231 test transactions",
                      "seconds": full_seed42["inference"]["median_seconds"],
                      "peak_process_rss_MiB": full_seed42["inference"]["peak_process_rss_MiB"]})

    xgb_val_frame, xgb_val = load_scores(root / "XGBoost", "validation")
    _, xgb_test = load_scores(root / "XGBoost", "test")
    neural_val_frame, neural_val = load_scores(root / "ablations" / "MambaJEPA_full" / "seed_42", "validation")
    _, neural_test = load_scores(root / "ablations" / "MambaJEPA_full" / "seed_42", "test")
    ssm_val_frame, ssm_val = load_scores(root / "Supervised_SSM" / "seed_42", "validation")
    _, ssm_test = load_scores(root / "Supervised_SSM" / "seed_42", "test")
    main_alpha = meta["models"]["XGBoost + MambaJEPA"]["validation_selected_alpha"]
    control_alpha = meta["models"]["XGBoost + supervised SSM"]["validation_selected_alpha"]

    fusion_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = main_alpha * xgb_val + (1 - main_alpha) * neural_val
        _ = main_alpha * xgb_test + (1 - main_alpha) * neural_test
        fusion_times.append(time.perf_counter() - start)
    fusion_resource = {"component": "Late fusion arithmetic for validation and test scores",
                       "seconds": float(np.median(fusion_times)), "peak_process_rss_MiB": None,
                       "measurement": "median of 100 in-memory repetitions; excludes prediction-file loading"}
    resources.append(fusion_resource)
    pd.DataFrame(resources).to_csv(root / "resource_usage.csv", index=False)
    meta["models"]["XGBoost + MambaJEPA"]["fusion_compute"] = fusion_resource

    main_test = main_alpha * xgb_test + (1 - main_alpha) * neural_test
    control_test = control_alpha * xgb_test + (1 - control_alpha) * ssm_test
    pairwise = paired_cluster_bootstrap(
        pd.read_csv(root / "XGBoost" / "test_predictions.csv.gz"),
        {"XGBoost + MambaJEPA": main_test, "XGBoost + supervised SSM": control_test, "XGBoost": xgb_test},
        meta["config"]["bootstrap_resamples"], meta["config"]["bootstrap_seed"],
    )
    meta["pairwise_customer_clustered_bootstrap"] = pairwise
    (root / "pairwise_bootstrap.json").write_text(json.dumps(pairwise, indent=2), encoding="utf-8")

    blind = meta["category_blind"]
    pd.DataFrame([
        {"model": "XGBoost", **{k: blind["XGBoost"][k] for k in ["validation_AUPRC", "test_ROC_AUC", "test_AUPRC", "test_F1", "test_MCC", "test_FPR"]}},
        {"model": "MambaJEPA neural seed 42", **{k: blind["MambaJEPA neural seed 42"][k] for k in ["validation_AUPRC", "test_ROC_AUC", "test_AUPRC", "test_F1", "test_MCC", "test_FPR"]}},
        {"model": "XGBoost + MambaJEPA", **{k: blind["XGBoost + MambaJEPA"][k] for k in ["validation_AUPRC", "test_ROC_AUC", "test_AUPRC", "test_F1", "test_MCC", "test_FPR"]},
         "validation_selected_alpha": blind["XGBoost + MambaJEPA"]["validation_selected_alpha"]},
    ]).to_csv(root / "category_blind_metrics.csv", index=False)

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        manifest[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(pairwise, indent=2))


if __name__ == "__main__":
    main()
