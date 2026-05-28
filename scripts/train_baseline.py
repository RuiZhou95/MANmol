from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manmol.baselines import build_baseline
from manmol.data import make_loaders
from manmol.seed import set_seed
from manmol.training import fit_regressor, load_yaml_config, select_device, write_audit_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(ROOT, args.config)
    set_seed(config["seed"])
    device = select_device(config["training"].get("device", "auto"))

    tokenizer_name = config.get("model", {}).get("smiles_model_name")
    if tokenizer_name is None:
        # tokenizer_name is read from config; fallback for backward compat only
        tokenizer_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    loaders, descriptor_columns = make_loaders(config, tokenizer)
    model = build_baseline(config, len(descriptor_columns)).to(device)

    output_dir = Path(config["training"]["output_dir"])
    write_audit_files(output_dir, config, descriptor_columns, device)

    if args.dry_run:
        from manmol.training import move_batch

        batch = move_batch(next(iter(loaders["train"])), device)
        predictions = model(batch)
        print({"baseline": config["baseline"], "predictions_shape": list(predictions.shape), "device": str(device)})
        return

    metrics = fit_regressor(model, loaders, config, device, output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
