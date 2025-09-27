import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pathlib import Path
import argparse
import json


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "resized" / "512"
TRAIN_VAL_CSV = PROC / "train_val.csv"
CONFIGS = ROOT / "configs"

LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.2
SEED = 42


def patient_table(df) -> pd.DataFrame:
    return df.groupby("patient_id", as_index=False)[LABELS].max()


def split_patients(pat_df: pd.DataFrame, val_ratio: float, seed: int):
    X = pat_df[["patient_id"]].to_numpy()
    Y = pat_df[LABELS].to_numpy(dtype=int)

    # Split into train vs temp using iterative stratification
    # We emulate a single split with a 2-fold KFold where one fold is ~val_ratio.
    # Compute fold sizes by shuffling and cutting, but keep iterative balance via KFold trick:
    n_splits = round(1 / val_ratio)  # e.g., 1/(1-0.2)=1.25 -> 1; fallback to 2
    n_splits = max(n_splits, 2)                 # ensure at least 2 folds
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(mskf.split(X, Y))
    # Take the first split: one fold as validation, the rest as train
    train_idx, val_idx = splits[0]
    train_pat = set(pat_df.iloc[train_idx]["patient_id"])
    val_pat   = set(pat_df.iloc[val_idx]["patient_id"])

    actual_val_ratio = len(val_idx) / len(Y)
    print(f"Requested val_ratio: {val_ratio:.3f}, Actual val_ratio: {actual_val_ratio:.3f}")

    # Ensure disjointness
    assert train_pat.isdisjoint(val_pat)
    return train_pat, val_pat


def compute_pos_weight(train_df: pd.DataFrame):
    N = len(train_df)
    P = train_df[LABELS].sum().to_numpy(dtype=float)
    P = np.clip(P, 1.0, None)         # avoid div-by-zero
    return ((N - P) / P).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_val_csv", default=TRAIN_VAL_CSV)
    ap.add_argument("--out_dir", default=PROC)
    ap.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    train_val_csv = Path(args.train_val_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_val_csv)
    required = {"patient_id", *LABELS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{train_val_csv} missing columns: {sorted(missing)}")

    pat_df = patient_table(df)
    train_pat, val_pat = split_patients(pat_df, args.val_ratio, args.seed)
    
    df["split"] = np.where(df["patient_id"].isin(train_pat), "train", "val")
    train_df = df[df.split == "train"].reset_index(drop=True).drop(["split"], axis=1)
    val_df   = df[df.split == "val"].reset_index(drop=True).drop(["split"], axis=1)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

    print("Counts:", {"train": len(train_df), "val": len(val_df)})
    print("Train prevalence:", train_df[LABELS].mean().round(4).to_dict())
    print("Val prevalence:",   val_df[LABELS].mean().round(4).to_dict())

    pos_weight = compute_pos_weight(train_df)
    (CONFIGS / "pos_weight.json").write_text(json.dumps({"labels": LABELS, "pos_weight": pos_weight}, indent=4))
    print("Wrote:", out_dir / "train.csv", out_dir / "val.csv", out_dir / "pos_weight.json")


if __name__ == "__main__":
    main()
