from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_id_split(ids, seed: int, train_size: float, val_size: float) -> pd.DataFrame:
    unique_ids = pd.Series(ids).drop_duplicates().sort_values().to_numpy()
    train_ids, temp_ids = train_test_split(unique_ids, train_size=train_size, random_state=seed, shuffle=True)
    val_fraction_of_temp = val_size / (1.0 - train_size)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_fraction_of_temp, random_state=seed, shuffle=True)

    rows = []
    for split, split_ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        rows.extend({"ID": int(mol_id), "split": split} for mol_id in split_ids)
    return pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor-csv", default="data/3-Opt_XGB_descriptor.csv")
    parser.add_argument("--output", default="data/splits/split_seed42.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--val-size", type=float, default=0.1)
    args = parser.parse_args()

    descriptor_path = Path(args.descriptor_csv)
    output_path = Path(args.output)
    df = pd.read_csv(descriptor_path, usecols=["ID"])
    split_df = make_id_split(df["ID"], args.seed, args.train_size, args.val_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)

    summary = {
        "seed": args.seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": 1.0 - args.train_size - args.val_size,
        "counts": split_df["split"].value_counts().to_dict(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
