import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    colors = {"MambaJEPA-AD": "#1f4e79", "XGBoost": "#c55a11", "CatBoost": "#548235", "LightGBM": "#8064a2"}
    paths = {"MambaJEPA-AD": root / "XGBoost_+_MambaJEPA", "XGBoost": root / "XGBoost",
             "CatBoost": root / "CatBoost", "LightGBM": root / "LightGBM"}
    plt.figure(figsize=(6.8, 4.4))
    for name, folder in paths.items():
        frame = pd.read_csv(folder / "test_predictions.csv.gz")
        precision, recall, _ = precision_recall_curve(frame.label, frame.score)
        plt.plot(recall, precision, lw=2, color=colors[name], label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.18)
    plt.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(out / "figure3_precision_recall.png", dpi=300)
    plt.close()

    table = pd.read_csv(root / "paper_tables" / "table4_category_blind.csv")
    x = np.arange(len(table))
    width = 0.34
    plt.figure(figsize=(6.8, 4.2))
    plt.bar(x - width/2, table["Full-feature AUPRC"], width, label="Full features", color="#1f4e79")
    plt.bar(x + width/2, table["Category-blind AUPRC"], width, label="Category blind", color="#9dc3e6")
    plt.xticks(x, ["XGBoost", "Neural\nbranch", "Fusion"])
    plt.ylabel("AUPRC")
    plt.ylim(0.70, 0.87)
    plt.grid(axis="y", alpha=0.18)
    plt.legend(frameon=False)
    for position, value in zip(x - width/2, table["Full-feature AUPRC"]):
        plt.text(position, value + 0.003, f"{value:.3f}", ha="center", fontsize=8)
    for position, value in zip(x + width/2, table["Category-blind AUPRC"]):
        plt.text(position, value + 0.003, f"{value:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "figure4_category_blind.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
