"""Derive Tables 3-8 directly from saved validation and test predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, matthews_corrcoef


def load(folder, split):
    return pd.read_csv(folder / f"{split}_predictions.csv.gz")


def threshold(y, score):
    precision, recall, values = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(values[int(np.argmax(f1))])


def report(validation, test, score_column="score"):
    t = threshold(validation.label, validation[score_column])
    predicted = test[score_column].to_numpy() >= t
    y = test.label.to_numpy()
    return {
        "AUC": roc_auc_score(y, test[score_column]),
        "AUPRC": average_precision_score(y, test[score_column]),
        "F1": f1_score(y, predicted),
        "MCC": matthews_corrcoef(y, predicted),
        "FPR": predicted[y == 0].mean(),
        "threshold": t,
    }


def fuse(xgb_validation, xgb_test, neural_validation, neural_test):
    grid = np.arange(21) / 20
    scores = [average_precision_score(xgb_validation.label, a * xgb_validation.score + (1-a) * neural_validation.score) for a in grid]
    alpha = float(grid[int(np.argmax(scores))])
    validation = xgb_validation.copy()
    test = xgb_test.copy()
    validation["score"] = alpha * xgb_validation.score + (1-alpha) * neural_validation.score
    test["score"] = alpha * xgb_test.score + (1-alpha) * neural_test.score
    return alpha, validation, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = Path(args.results)
    meta = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    tables = root / "paper_tables"
    tables.mkdir(exist_ok=True)

    table3_names = ["XGBoost + MambaJEPA", "XGBoost", "CatBoost", "LightGBM", "Random Forest",
                    "Balanced MLP", "Supervised LSTM", "Supervised SSM", "Supervised Transformer",
                    "XGBoost + supervised SSM"]
    rows = []
    for name in table3_names:
        item = meta["models"][name]
        rows.append({"Method": name, "AUC": item["test_ROC_AUC"], "AUPRC": item["test_AUPRC"],
                     "F1": item["test_F1"], "MCC": item["test_MCC"], "FPR": item["test_FPR"],
                     "validation_threshold": item["validation_selected_F1_threshold"],
                     "validation_alpha": item.get("validation_selected_alpha")})
    pd.DataFrame(rows).to_csv(tables / "table3_baselines.csv", index=False)

    blind = pd.read_csv(root / "category_blind_metrics.csv")
    full = {row["Method"]: row for row in rows}
    table4 = []
    for blind_name, full_name in [("XGBoost", "XGBoost"), ("MambaJEPA neural seed 42", "MambaJEPA full seed 42"),
                                  ("XGBoost + MambaJEPA", "XGBoost + MambaJEPA")]:
        blind_value = float(blind.loc[blind.model == blind_name, "test_AUPRC"].iloc[0])
        if full_name == "MambaJEPA full seed 42":
            full_value = meta["ablations"]["MambaJEPA full"]["runs"][0]["test_AUPRC"]
        else:
            full_value = full[full_name]["AUPRC"]
        table4.append({"Model": blind_name, "Full-feature AUPRC": full_value,
                       "Category-blind AUPRC": blind_value, "Change": blind_value - full_value})
    pd.DataFrame(table4).to_csv(tables / "table4_category_blind.csv", index=False)

    table5 = []
    for name, group in meta["ablations"].items():
        runs = group["runs"]
        row = {"Configuration": name}
        for source, label in [("test_ROC_AUC", "AUC"), ("test_AUPRC", "AUPRC"), ("test_F1", "F1")]:
            values = np.asarray([run[source] for run in runs])
            row[f"{label} mean"] = values.mean()
            row[f"{label} sample SD"] = values.std(ddof=1)
        table5.append(row)
    pd.DataFrame(table5).to_csv(tables / "table5_ablations.csv", index=False)

    xgb_v, xgb_t = load(root / "XGBoost", "validation"), load(root / "XGBoost", "test")
    main_v, main_t = load(root / "XGBoost_+_MambaJEPA", "validation"), load(root / "XGBoost_+_MambaJEPA", "test")
    normal = main_v.loc[main_v.label == 0, "score"].to_numpy()
    u = np.quantile(normal, 0.95)
    excess = normal[normal > u] - u
    shape, _, scale = genpareto.fit(excess, floc=0)
    table6 = []
    for risk in [0.001, 0.0025, 0.005, 0.01]:
        tail = len(excess) / len(normal)
        cutoff = u + scale * np.log(tail / risk) if abs(shape) < 1e-8 else u + scale / shape * ((tail / risk) ** shape - 1)
        predicted = main_t.score.to_numpy() >= cutoff
        y = main_t.label.to_numpy()
        tp = int(((predicted == 1) & (y == 1)).sum())
        fp = int(((predicted == 1) & (y == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(int((y == 1).sum()), 1)
        table6.append({"Target normal-tail risk": risk, "Threshold": cutoff, "Precision": precision,
                       "Recall": recall, "F1": 2 * precision * recall / max(precision + recall, 1e-15),
                       "Observed FPR": fp / max(int((y == 0).sum()), 1)})
    pd.DataFrame(table6).to_csv(tables / "table6_evt.csv", index=False)

    table7 = []
    for seed in meta["config"]["seeds"]:
        nv = load(root / "ablations" / "MambaJEPA_full" / f"seed_{seed}", "validation")
        nt = load(root / "ablations" / "MambaJEPA_full" / f"seed_{seed}", "test")
        alpha, fv, ft = fuse(xgb_v, xgb_t, nv, nt)
        r = report(fv, ft)
        table7.append({"Neural seed": seed, "Neural AUPRC": average_precision_score(nt.label, nt.score),
                       "Validation-selected alpha": alpha, "Fused AUPRC": r["AUPRC"], "Fused AUC": r["AUC"],
                       "Fused F1": r["F1"], "Validation-selected threshold": r["threshold"]})
    pd.DataFrame(table7).to_csv(tables / "table7_seed_stability.csv", index=False)

    model_frames = {
        "MambaJEPA-AD": main_t,
        "XGBoost": xgb_t,
        "CatBoost": load(root / "CatBoost", "test"),
        "LightGBM": load(root / "LightGBM", "test"),
    }
    table8 = []
    for name, frame in model_frames.items():
        y = frame.label.to_numpy()
        score = frame.score.to_numpy()
        row = {"Method": name}
        for fraction in [0.005, 0.01, 0.02]:
            count = int(np.ceil(len(frame) * fraction))
            selected = np.argsort(-score, kind="stable")[:count]
            tp = int(y[selected].sum())
            fp = count - tp
            row[f"reviewed_{fraction:.3f}"] = count
            row[f"precision_{fraction:.3f}"] = tp / count
            row[f"recall_{fraction:.3f}"] = tp / int(y.sum())
            row[f"FPR_{fraction:.3f}"] = fp / int((y == 0).sum())
        table8.append(row)
    pd.DataFrame(table8).to_csv(tables / "table8_review_budgets.csv", index=False)

    windows = []
    for lo, hi in [(144, 155), (156, 167), (168, 179)]:
        mask = main_t.step.between(lo, hi)
        windows.append({"Steps": f"{lo}-{hi}", "MambaJEPA fusion AUPRC": average_precision_score(main_t.label[mask], main_t.score[mask]),
                        "XGBoost AUPRC": average_precision_score(xgb_t.label[mask], xgb_t.score[mask])})
    pd.DataFrame(windows).to_csv(tables / "temporal_windows.csv", index=False)

    print((tables / "table3_baselines.csv").read_text(encoding="utf-8"))
    print((tables / "table5_ablations.csv").read_text(encoding="utf-8"))
    print((tables / "table7_seed_stability.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
