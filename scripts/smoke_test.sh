#!/bin/bash
# Smoke test: unit tests + dry-runs for all key configs
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

echo "=== Unit Tests ==="
pytest -q tests/

echo ""
echo "=== MANmol dry-run ==="
python scripts/train_manmol.py --config configs/manmol_clean.yaml --dry-run

echo ""
echo "=== Descriptor MLP dry-run ==="
python scripts/train_baseline.py --config configs/baseline_descriptor_mlp.yaml --dry-run

echo ""
echo "=== Graph GINE dry-run ==="
python scripts/train_baseline.py --config configs/baseline_graph_gine.yaml --dry-run

echo ""
echo "=== SMILES ChemBERTa dry-run ==="
python scripts/train_baseline.py --config configs/baseline_smiles_chemberta.yaml --dry-run

echo ""
echo "=== All smoke tests passed ==="

