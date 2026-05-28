from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manmol.data import make_loaders
from manmol.metrics import regression_metrics
from manmol.models import ALL_MODALITIES, MANmolClean
from manmol.seed import set_seed
from manmol.training import write_audit_files


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(ROOT / candidate)


def select_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "ids": batch["ids"].to(device),
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "descriptors": batch["descriptors"].to(device),
        "graphs": batch["graphs"].to(device),
        "targets": batch["targets"].to(device),
    }


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device, gate_entropy_lambda: float = 0.0) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions, weights = model(batch["input_ids"], batch["attention_mask"], batch["descriptors"], batch["graphs"])
        mse_loss = criterion(predictions, batch["targets"])
        loss = mse_loss
        if gate_entropy_lambda and model.fusion in ("gated", "channel_gate"):
            if model.fusion == "channel_gate" and model._last_gate is not None:
                gate = model._last_gate
                entropy = -(gate * gate.clamp_min(1e-8).log() + (1 - gate) * (1 - gate).clamp_min(1e-8).log()).mean()
            else:
                entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
            loss = loss - gate_entropy_lambda * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += mse_loss.item() * batch["targets"].numel()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device: torch.device) -> tuple[list[int], list[float], list[float], list[list[float]]]:
    model.eval()
    ids: list[int] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    weights: list[list[float]] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        predictions, modal_weights = model(batch["input_ids"], batch["attention_mask"], batch["descriptors"], batch["graphs"])
        ids.extend(batch["ids"].detach().cpu().tolist())
        y_true.extend(batch["targets"].detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        weights.extend(modal_weights.detach().cpu().tolist())
    return ids, y_true, y_pred, weights


def expand_weights(weights: list[list[float]], modalities: list[str]) -> list[list[float]]:
    expanded = []
    for modal_weight in weights:
        row = [0.0] * len(ALL_MODALITIES)
        for index, modality in enumerate(modalities):
            row[ALL_MODALITIES.index(modality)] = modal_weight[index]
        expanded.append(row)
    return expanded


def write_predictions(path: Path, ids, y_true, y_pred, weights, modalities: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = expand_weights(weights, modalities)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "target", "prediction", "weight_smiles", "weight_descriptor", "weight_graph"])
        for mol_id, target, prediction, modal_weight in zip(ids, y_true, y_pred, weights):
            writer.writerow([mol_id, target, prediction, *modal_weight])


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["data"]["descriptor_csv"] = resolve_path(config["data"]["descriptor_csv"])
    config["data"]["graph_pkl"] = resolve_path(config["data"]["graph_pkl"])
    config["data"]["split_csv"] = resolve_path(config["data"]["split_csv"])
    config["training"]["output_dir"] = resolve_path(config["training"]["output_dir"])
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manmol_clean.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = select_device(config["training"].get("device", "auto"))

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["smiles_model_name"])
    loaders, descriptor_columns = make_loaders(config, tokenizer)
    model = MANmolClean(
        smiles_model_name=config["model"]["smiles_model_name"],
        descriptor_size=len(descriptor_columns),
        node_input_dim=config["model"].get("node_input_dim", 3),
        hidden_size=config["model"]["hidden_size"],
        gnn_layers=config["model"]["gnn_layers"],
        gnn_edge_dim=config["model"]["gnn_edge_dim"],
        dropout=config["model"]["dropout"],
        freeze_smiles_encoder=config["model"].get("freeze_smiles_encoder", False),
        modalities=config["model"].get("modalities"),
        fusion=config["model"].get("fusion", "gated"),
        gate_temperature=config["model"].get("gate_temperature", 1.0),
        modality_layer_norm=config["model"].get("modality_layer_norm", False),
        modality_dropout=config["model"].get("modality_dropout", 0.0),
    ).to(device)

    output_dir = Path(config["training"]["output_dir"])
    write_audit_files(output_dir, config, descriptor_columns, device)

    if args.dry_run:
        batch = move_batch(next(iter(loaders["train"])), device)
        predictions, weights = model(batch["input_ids"], batch["attention_mask"], batch["descriptors"], batch["graphs"])
        print({
            "predictions_shape": list(predictions.shape),
            "weights_shape": list(weights.shape),
            "modalities": model.modalities,
            "fusion": model.fusion,
            "device": str(device),
        })
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    criterion = nn.MSELoss()
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0
    log_path = output_dir / "train_log.csv"

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_r2", "val_mse", "val_rmse", "val_mae"])
        for epoch in range(1, config["training"]["epochs"] + 1):
            train_loss = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                criterion,
                device,
                gate_entropy_lambda=config["training"].get("gate_entropy_lambda", 0.0),
            )
            _, val_true, val_pred, _ = predict(model, loaders["val"], device)
            val_metrics = regression_metrics(val_true, val_pred)
            writer.writerow([epoch, train_loss, val_metrics["r2"], val_metrics["mse"], val_metrics["rmse"], val_metrics["mae"]])
            handle.flush()
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_rmse={val_metrics['rmse']:.6f} val_r2={val_metrics['r2']:.6f}")

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= config["training"]["patience"]:
                    break

    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    all_metrics = {"best_epoch": best_epoch}
    for split in ("train", "val", "test"):
        ids, y_true, y_pred, weights = predict(model, loaders[split], device)
        all_metrics[split] = regression_metrics(y_true, y_pred)
        write_predictions(output_dir / f"predictions_{split}.csv", ids, y_true, y_pred, weights, model.modalities)
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
