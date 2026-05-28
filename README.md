# MANmol: Multimodal Attention Network for Molecules

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/)

**MANmol** (Multimodal Attention Network for Molecules) is an advanced deep learning framework for molecular adsorption energy prediction, specifically designed for green lubricant additive discovery. This project integrates multimodal feature fusion, adaptive attention weighting, and transfer learning techniques to address data scarcity and accurately characterize molecular adsorption behavior.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [✨ Core Features](#-core-features)
- [🏗️ Architecture Design](#️-architecture-design)
- [📊 Datasets](#-datasets)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🔧 Installation Guide](#-installation-guide)
- [💻 Usage Examples](#-usage-examples)
- [📈 Experimental Results](#-experimental-results)
- [🤝 Contribution Guidelines](#-contribution-guidelines)
- [📄 License](#-license)
- [📚 Citation](#-citation)
- [📞 Contact Us](#-contact-us)

## Project Overview

MANmol predicts molecular adsorption energy on carbon steel surfaces for green lubricant additive discovery. It fuses three complementary molecular representations:

- **Molecular Graph (GNN)**: GINEConv on RDKit-rich atom/bond features — captures local topology and bond-level interactions
- **SMILES Sequence (Transformer)**: ChemBERTa encoder — captures long-range sequential patterns
- **Molecular Descriptors (MLP)**: XGBoost-selected physicochemical descriptors — captures global molecular properties

### Research Background
- **Problem**: Traditional lubricant additive discovery processes are time-consuming and labor-intensive, requiring extensive experimental validation
- **Solution**: Utilize machine learning methods for high-throughput virtual screening
- **Innovation**: Multimodal attention network integrates multiple molecular representations to improve prediction accuracy

## ✨ Core Features

### 1. **Multimodal Feature Fusion**
   - Integrates three modalities: molecular graphs (GNN), SMILES sequences (Transformer), and molecular descriptors (MLP)
   - Adaptive attention mechanism dynamically weights the importance of different modalities

### 2. **Automated Feature Engineering**
   - Complete feature calculation pipeline (2D/3D descriptors, GAFF2 force field descriptors)
   - Feature selection methods based on statistics and machine learning
   - SMILES data augmentation and molecular graph data augmentation

### 3. **Pre-trained Model Support**
   - Supports fine-tuning of multiple pre-trained models: ChemBERTa, BART, GAT, GCN, GIN, GraphSAGE, MoLFormer, MPNN
   - LoRA (Low-Rank Adaptation) efficient fine-tuning technology
   - SMILES mask pre-trained models

### 4. **Symbolic Regression Analysis**
   - Symbolic regression based on PySR to discover interpretable mathematical relationships
   - Provides physically meaningful adsorption energy prediction formulas

### 5. **High-Performance Computing Optimization**
   - Supports GPU cluster computing environments
   - Distributed training and inference optimization
   - Large-scale dataset processing capability (376 million organic compounds)

## 🏗️ Architecture Design
```
Input Molecule
    │
    ├──→ SMILES ──→ ChemBERTa ──→ Self-Attn ──→ [h_smiles]
    │
    ├──→ Descriptors ──→ MLP ────────────────→ [h_descriptor]
    │
    └──→ RDKit Graph ──→ GINEConv ──→ Pool ──→ [h_graph]
                              │
                    ┌─────────┴─────────┐
                    │   Fusion Module   │
                    │  concat | gated   │
                    │  channel_gate |   │
                    │  cross_attend     │
                    └────────┬──────────┘
                             │
                        MLP Regressor
                             │
                       Prediction (y)
```
### Overall Architecture

![MANmol architecture](./data/cover_letter.png)

*Figure: MANmol multimodal attention network architecture integrating molecular graphs, SMILES sequences, and molecular descriptors through adaptive attention fusion.*

### Technology Stack
- **Deep Learning Framework**: PyTorch, PyTorch Geometric
- **Natural Language Processing**: Hugging Face Transformers([ChemBERTa](https://huggingface.co/DeepChem/ChemBERTa-100M-MLM))
- **Feature Engineering**: RDKit, Mordred, OpenBabel
- **Symbolic Regression**: PySR
- **Efficient Fine-tuning**: PEFT (LoRA)
- **Data Processing**: Pandas, NumPy, Scikit-learn

## 📊 Datasets

### 1. **AEdata Dataset**
   - **Size**: 13,320 organic small molecules
   - **Content**: Adsorption energy of molecules on carbon steel surfaces
   - **Source**: All-atom molecular dynamics simulations
   - **Format**: CSV file containing ID, SMILES, adsorption energy, and other fields

### 2. **OCSmi Dataset** (Requires separate acquisition)
   - **Size**: 376 million organic compound SMILES
   - **Content**: Large-scale organic compound library
   - **Access**: Contact the corresponding author
   - **Purpose**: Pre-training and transfer learning

### 3. **Augmented Datasets**
   - **Molecular Graph Augmentation**: 13,320 → 91,241 samples (through rotation, flipping, etc.)
   - **SMILES Augmentation**: Non-canonical SMILES representations
   - **Multimodal Combination**: Random combination of different modality features

## 🚀 Quick Start

### Environment Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- At least 16GB RAM (for processing large-scale datasets)

### Basic Usage
```bash
# Clone the repository
git clone https://github.com/RuiZhou95/MANmol.git
cd MANmol

# Create conda environment
conda create -n manmol python=3.9
conda activate manmol

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Run tests
PYTHONPATH=. pytest -q tests/

# Dry-run to verify setup
python scripts/train_manmol.py --config configs/manmol_clean.yaml --dry-run

# Train the recommended model (descriptor + graph concat)
python scripts/train_manmol.py --config configs/fusion_ablation/dg_concat_rdkit.yaml

# Train with all three modalities
python scripts/train_manmol.py --config configs/fusion_ablation/all_concat_rdkit.yaml

# Train a baseline
python scripts/train_baseline.py --config configs/baseline_descriptor_mlp.yaml
```

The ChemBERTa encoder is downloaded automatically from HuggingFace ([ChemBERTa](https://huggingface.co/DeepChem/ChemBERTa-100M-MLM)) on first use.

## 📁 Project Structure

```
├── manmol/                          # Clean pipeline (core package)
│   ├── models.py                    # MANmolClean model + fusion variants
│   ├── data.py                      # Data loading, splitting, batching
│   ├── baselines.py                 # Single-modality baseline models
│   ├── graph_features.py            # RDKit rich graph feature extraction
│   ├── metrics.py                   # Regression metrics (R2, RMSE, MAE)
│   ├── training.py                  # Training utilities
│   └── seed.py                      # Reproducibility
│
├── scripts/                         # Training and utility scripts
│   ├── train_manmol.py              # Train multimodal model
│   ├── train_baseline.py            # Train single-modality baseline
│   ├── make_splits.py               # Generate ID-level splits
│   ├── audit_graph_data.py          # Graph data quality audit
│   └── smoke_test.sh                # End-to-end validation
│
├── configs/                         # YAML configuration files
│   ├── manmol_clean.yaml            # Default MANmol (legacy graph)
│   ├── manmol_rdkit_rich.yaml       # MANmol with RDKit rich graph
│   ├── baseline_*.yaml              # Single-modality baseline configs
│   ├── fusion_ablation/             # Fusion variant experiments
│   │   ├── dg_concat_rdkit.yaml     # Descriptor + Graph concat (recommended)
│   │   ├── all_concat_rdkit.yaml    # All-3 concat (best overall)
│   │   ├── all_channel_gate_rdkit.yaml
│   │   ├── all_cross_attend_rdkit.yaml
│   │   └── ...
│   └── graph_debug/                 # Graph debugging experiments
│
├── tests/                           # Unit tests
│   ├── test_forward_shapes.py
│   ├── test_graph_batch.py
│   └── test_split_no_leakage.py
│
├── data/                            # Data files
│   ├── 3-Opt_XGB_descriptor.csv
│   ├── all_data_ID-smiles-graph-13320.pkl
│   └── splits/split_seed42.csv
│
├── MANmol/                          # Legacy code (original experiments)
├── Fine-tuning_multiple_models/      # Legacy baseline scripts
├── SMILES_Mask_pre-trained/          # SMILES pre-training scripts
├── Automated_feature_engineering/    # Feature engineering pipeline
├── symbolic regression/              # Symbolic regression analysis
│
├── outputs_fusion_ablation/          # Experiment outputs (gitignored)
├── outputs_graph_debug/              # Graph debug outputs (gitignored)
│
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔧 Installation Guide
### 1. Dependency Installation
```bash
# PyTorch with CUDA
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch_geometric

# Core dependencies
pip install transformers pandas numpy scikit-learn rdkit pyyaml tqdm

# Optional
pip install xgboost pytest
```

### 3. Verify Installation
```bash
PYTHONPATH=. python -c "
import torch
from manmol.models import MANmolClean
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print('MANmol OK')
"
```

## 💻 Usage Examples

### Train a Model
```bash
# Descriptor + Graph concat (recommended — best performance without SMILES)
python scripts/train_manmol.py --config configs/fusion_ablation/dg_concat_rdkit.yaml
```

### Train with Custom Modalities and Fusion
```bash
# All three modalities with channel gate fusion
python scripts/train_manmol.py --config configs/fusion_ablation/all_channel_gate_rdkit.yaml
```

### Programmatic Usage
```python
from manmol.models import MANmolClean

model = MANmolClean(
    smiles_model_name="DeepChem/ChemBERTa-77M-MLM",
    descriptor_size=33,
    node_input_dim=18,
    hidden_size=128,
    gnn_layers=3,
    gnn_edge_dim=12,
    modalities=["descriptor", "graph"],    # skip SMILES
    fusion="concat",                        # simple concat
    modality_dropout=0.1,                   # optional regularization
)
```

### Run All Smoke Tests
```bash
bash scripts/smoke_test.sh
```

## Experimental Results

All results below use **strict ID-level split** (seed=42), **RDKit rich graph**, and **single test evaluation** after best validation checkpoint.

### Key Findings
1. **Multimodal fusion significantly improves performance**: MANmol improves R² by 5-9% compared to single-modal models
2. **Attention mechanism is effective**: Adaptive attention can dynamically adjust the importance of different modalities
3. **Simple concat fusion is sufficient**. Neither scalar gating, channel-wise gating, nor cross-modal attention outperforms concatenation + MLP.
4. **Per-modality LayerNorm degrades performance** (R2 = 0.9397 vs 0.9515), likely because it removes informative scale differences between modality embeddings.
5. **Data augmentation shows clear effects**: Molecular graph augmentation increases training samples by 6.8x, improving model generalization
6. **Symbolic regression provides interpretability**: Discovers mathematical relationships between adsorption energy and molecular properties like polarity and size

## Fusion Variants

| Config | Modalities | Fusion | Notes |
|---|---|---|---|
| `dg_concat_rdkit.yaml` | D+G | concat | Recommended (best without SMILES) |
| `all_concat_rdkit.yaml` | S+D+G | concat | Best overall (marginal SMILES benefit) |
| `all_channel_gate_rdkit.yaml` | S+D+G | channel_gate | Dimension-wise gating |
| `all_cross_attend_rdkit.yaml` | S+D+G | cross_attend | Cross-modal attention |
| `all_concat_dropout_rdkit.yaml` | S+D+G | concat + dropout | Modality dropout (p=0.2) |
| `all_gated_rdkit.yaml` | S+D+G | gated | Original scalar gating |
| `dg_gated_rdkit.yaml` | D+G | gated | Pairwise gated |
| `ds_gated_rdkit.yaml` | S+D | gated | SMILES + descriptor |
| `sg_gated_rdkit.yaml` | S+G | gated | SMILES + graph |

All configs in `configs/fusion_ablation/`. See `v1_review.md` for detailed analysis.

## 🤝 Contribution Guidelines

We welcome and appreciate all forms of contributions!

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Create a Pull Request**

### Contribution Areas
- Code optimization and improvements
- Documentation enhancement and translation
- New features and model additions
- Bug fixes and issue reporting
- Performance testing and benchmarking

### Development Standards
- Follow PEP 8 code style
- Add appropriate comments and documentation
- Write unit tests
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use MANmol in your research, please cite our paper:

```bibtex
@article{###,
  title={###},
  author={###},
  journal={###},
  volume={###},
  pages={###},
  year={###},
  publisher={###}
}
```

## 📞 Contact Us

### Project Lead
- **Rui Zhou** - Main Developer
- **Email**: zhourui@licp.cas.cn
- **GitHub**: [@RuiZhou95](https://github.com/RuiZhou95)

### Related Projects
1. **VIIInfo** (Viscosity Index Improver Informatics)
   - Link: https://github.com/RuiZhou95/Viscosity-Index-Improvers
   - Description: Machine learning pipeline for high-performance viscosity index improver polymers

2. **iFEQ** (interpretable Feature-Engineered QSAR)
   - Link: https://github.com/RuiZhou95/ML4IL
   - Description: End-to-end toolkit for interpretable feature-engineered QSAR

### Issue Reporting
- Use GitHub Issues to report problems
- Contact via email for dataset access permissions
- Join our academic discussion group

---

## 🎯 Future Plans

### Short-term Goals (2026)
- [ ] Release pre-trained models to Hugging Face Hub
- [ ] Develop web interface and API services
- [ ] Add more molecular representation modalities (3D structures, fingerprints, etc.)

### Medium-to-long-term Goals (2027+)
- [ ] Extend to other molecular property prediction tasks
- [ ] Integrate active learning framework
- [ ] Develop cloud deployment solutions
- [ ] Build community and user ecosystem

---

**⭐ If you find this project helpful, please give us a Star!** ⭐

**🙏 Thank you for your interest and support of the MANmol project!**

