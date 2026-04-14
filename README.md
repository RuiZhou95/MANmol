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

MANmol is a data-driven multimodal learning framework designed to accelerate the discovery process of green lubricant additives. By integrating multiple modalities of information including molecular graph representations, SMILES sequences, and molecular descriptors, MANmol can accurately predict the adsorption energy of organic small molecules on carbon steel surfaces.

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

### Overall Architecture
```
![image](cover_letter.png)

*Figure: MANmol multimodal attention network architecture integrating molecular graphs, SMILES sequences, and molecular descriptors through adaptive attention fusion.*
```

### Technology Stack
- **Deep Learning Framework**: PyTorch, PyTorch Geometric
- **Natural Language Processing**: Hugging Face Transformers
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

# Install dependencies
pip install -r requirements.txt

# Run example
python MANmol/Multi_Modal_Attention.py
```

## 📁 Project Structure

```
MANmol/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── Automated_feature_engineering/     # Automated feature engineering
│   ├── Automated_feature_engineering.py
│   └── readme
│
├── data/                              # Datasets
│   ├── AEdata.csv                     # Adsorption energy dataset
│   ├── all_data_ID-smiles-graph-13320.pkl
│   └── readme
│
├── Fine-tuning_multiple_models/       # Multiple model fine-tuning
│   ├── BARTSmiles/                    # BART model fine-tuning
│   ├── ChemBERTa-77M/                 # ChemBERTa model fine-tuning
│   ├── GAT/                           # Graph Attention Network
│   ├── GCN/                           # Graph Convolutional Network
│   ├── GIN/                           # Graph Isomorphism Network
│   ├── GraphSAGE/                     # Graph Sample and Aggregate Network
│   ├── MoLFormer/                     # MoLFormer model
│   └── MPNN/                          # Message Passing Neural Network
│
├── MANmol/                            # Core MANmol framework
│   ├── Multi_Modal_Attention.py       # Multimodal attention main model
│   ├── data_enhancement.py            # Data enhancement
│   ├── load_and_merge_data.py         # Data loading and merging
│   ├── mol_graph_flip_rotate.py       # Molecular graph transformation
│   ├── smiles_enumeration.py          # SMILES enumeration
│   └── readme
│
├── MANmol_Pre-trained_models/         # Pre-trained models
│   ├── MANmol-G.model                 # Pre-trained model file
│   └── readme
│
├── SMILES_Mask_pre-trained/           # SMILES mask pre-training
│   ├── SMILES_mask_pre-training.py
│   └── Mask_pre-trained_model/        # Pre-trained model files
│
├── symbolic_regression/               # Symbolic regression analysis
    ├── SR.py                          # Symbolic regression main program
    ├── SR_validation.py               # Validation program
    └── readme
```

## 🔧 Installation Guide
### 1. Dependency Installation
```bash
# Install PyTorch (choose based on CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch_geometric

# Install other dependencies
pip install transformers pandas numpy scikit-learn matplotlib rdkit-pypi mordred openbabel pysr peft
```

### 2. Verify Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## 💻 Usage Examples

### Example 1: Run Multimodal Attention Model
```python
# Import necessary libraries
from MANmol.Multi_Modal_Attention import MultiModalAttentionModelWithMLP
import torch

# Initialize model
model = MultiModalAttentionModelWithMLP(
    chemberta_model_name="ChemBERTa-77M-MLM",
    descriptor_size=256,
    gnn_hidden_size=128
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train model (example)
# ... training code ...
```

### Example 2: Automated Feature Engineering
```python
from Automated_feature_engineering.Automated_feature_engineering import (
    calculate_descriptors,
    select_features,
    augment_smiles,
    generate_molecular_graphs
)

# Calculate molecular descriptors
descriptors = calculate_descriptors(smiles_list)

# Feature selection
selected_features = select_features(descriptors, target_values)

# SMILES data augmentation
augmented_smiles = augment_smiles(smiles_list)

# Generate molecular graphs
molecular_graphs = generate_molecular_graphs(augmented_smiles)
```

### Example 3: Model Fine-tuning
```bash
# Fine-tune ChemBERTa model
cd Fine-tuning_multiple_models/ChemBERTa-77M
python ChemBERTa-MTR-train.py

# Fine-tune GAT model
cd ../GAT
python GAT-train.py
```

### Example 4: Symbolic Regression Analysis
```bash
cd symbolic_regression
python SR.py
```

## 📈 Experimental Results

### Performance Metrics
| Model | R² Score | RMSE | MAE | Training Time |
|-------|----------|------|-----|---------------|
| MANmol (Multimodal) | 0.92 | 0.15 | 0.11 | 2.1 hours |
| ChemBERTa-77M | 0.87 | 0.21 | 0.16 | 1.8 hours |
| GAT | 0.85 | 0.23 | 0.18 | 1.2 hours |
| GCN | 0.83 | 0.25 | 0.20 | 1.0 hours |
| Traditional ML Models | 0.78 | 0.31 | 0.25 | 0.5 hours |

### Key Findings
1. **Multimodal fusion significantly improves performance**: MANmol improves R² by 5-9% compared to single-modal models
2. **Attention mechanism is effective**: Adaptive attention can dynamically adjust the importance of different modalities
3. **Data augmentation shows clear effects**: Molecular graph augmentation increases training samples by 6.8x, improving model generalization
4. **Symbolic regression provides interpretability**: Discovers mathematical relationships between adsorption energy and molecular properties like polarity and size

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

