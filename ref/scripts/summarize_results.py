import glob, json, os, statistics, csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

OUT = "outputs/summary"
os.makedirs(OUT, exist_ok=True)
rows = []
for p in glob.glob("outputs/*/metrics.json"):
    d = json.load(open(p))
    a = d["args"]
    rows.append(
        {
            "run": os.path.dirname(p),
            "model": a["model"],
            "augmentation": a["aug"],
            "loss": a["loss"],
            "seed": a["seed"],
            **{
                k: d[k]
                for k in [
                    "accuracy",
                    "macro_f1",
                    "macro_precision",
                    "macro_recall",
                    "weighted_f1",
                    "balanced_accuracy",
                    "ordinal_mae",
                    "within_one",
                ]
            },
        }
    )
with open(OUT + "/all_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
final = []
for m in ["simple", "complex", "mobile"]:
    x = [
        r
        for r in rows
        if r["model"] == m and r["augmentation"] == "A1" and r["loss"] == "weighted"
    ]
    z = {"model": m, "n": len(x)}
    for k in [
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "weighted_f1",
        "balanced_accuracy",
        "ordinal_mae",
        "within_one",
    ]:
        v = [r[k] for r in x]
        z[k + "_mean"] = statistics.mean(v)
        z[k + "_std"] = statistics.stdev(v) if len(v) > 1 else 0
    final.append(z)
with open(OUT + "/final_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=final[0])
    w.writeheader()
    w.writerows(final)
for m in ["simple", "complex", "mobile"]:
    x = next(
        r
        for r in rows
        if r["model"] == m
        and r["seed"] == 42
        and r["augmentation"] == "A1"
        and r["loss"] == "weighted"
    )
    d = json.load(open(x["run"] + "/metrics.json"))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        np.array(d["confusion_matrix"]),
        display_labels=["empty", "low", "medium", "high", "traffic_jam"],
    ).plot(ax=ax, colorbar=False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/confusion_{m}.png", dpi=160)
    plt.close()
    h = json.load(open(x["run"] + "/history.json"))
    plt.plot([q["macro_f1"] for q in h], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Macro-F1")
    plt.title(f"{m}: seed 42")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/curve_{m}.png", dpi=160)
    plt.close()
print(json.dumps(final, indent=2))
