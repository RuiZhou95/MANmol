from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from tqdm import tqdm

from manmol.metrics import regression_metrics


def resolve_project_path(root: Path, path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(root / candidate)


def resolve_common_paths(root: Path, config: dict) -> dict:
    config["data"]["descriptor_csv"] = resolve_project_path(root, config["data"]["descriptor_csv"])
    config["data"]["graph_pkl"] = resolve_project_path(root, config["data"]["graph_pkl"])
    config["data"]["split_csv"] = resolve_project_path(root, config["data"]["split_csv"])
    config["training"]["output_dir"] = resolve_project_path(root, config["training"]["output_dir"])
    return config


def load_yaml_config(root: Path, path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return resolve_common_paths(root, config)


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


def write_audit_files(output_dir: Path, config: dict, descriptor_columns: list[str], device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "descriptor_columns.json").write_text(json.dumps(descriptor_columns, indent=2), encoding="utf-8")
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }
    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_name"] = torch.cuda.get_device_name(device)
    (output_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch)
        loss = criterion(predictions, batch["targets"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * batch["targets"].numel()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device: torch.device) -> tuple[list[int], list[float], list[float]]:
    model.eval()
    ids: list[int] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        predictions = model(batch)
        ids.extend(batch["ids"].detach().cpu().tolist())
        y_true.extend(batch["targets"].detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
    return ids, y_true, y_pred


def write_predictions(path: Path, ids, y_true, y_pred) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "target", "prediction"])
        for mol_id, target, prediction in zip(ids, y_true, y_pred):
            writer.writerow([mol_id, target, prediction])


def fit_regressor(model, loaders, config: dict, device: torch.device, output_dir: Path) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    criterion = nn.MSELoss()
    best_val_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0

    with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_r2", "val_mse", "val_rmse", "val_mae"])
        for epoch in range(1, config["training"]["epochs"] + 1):
            train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
            _, val_true, val_pred = predict(model, loaders["val"], device)
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
        ids, y_true, y_pred = predict(model, loaders[split], device)
        all_metrics[split] = regression_metrics(y_true, y_pred)
        write_predictions(output_dir / f"predictions_{split}.csv", ids, y_true, y_pred)
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return all_metrics
