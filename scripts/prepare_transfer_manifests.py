"""Create transfer-learning manifests by excluding verified Category-A images only."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "data" / "manifests"
PAIR_FILE = PROJECT_DIR / "outputs" / "near_duplicate_audit" / "near_duplicate_pairs.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "transfer_final_manifests"


def main():
    pairs = pd.read_csv(PAIR_FILE)
    confirmed = pairs.loc[
        pairs["category"].eq("A_confirmed_near_duplicate"),
        ["image_a", "image_b"],
    ]
    excluded = set(confirmed["image_a"]) | set(confirmed["image_b"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for split in ("train", "val", "test"):
        source = pd.read_csv(SOURCE_DIR / f"{split}.csv")
        derived = source.loc[~source["image_path"].isin(excluded)].copy()
        derived.to_csv(OUTPUT_DIR / f"{split}.csv", index=False)
        summary.append(
            {"split": split, "original": len(source), "removed_A": len(source) - len(derived), "final": len(derived)}
        )

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "summary.csv", index=False)
    pd.DataFrame({"image_path": sorted(excluded)}).to_csv(OUTPUT_DIR / "excluded_category_A.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
