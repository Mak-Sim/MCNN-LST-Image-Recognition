# Towards Lightweight Image Recognition: Multi-Column Convolutional Neural Networks Based on Learned Separable Transform

[![DOI](https://img.shields.io/badge/DOI-10.1049/ipr2.70362-blue)](https://doi.org/10.1049/ipr2.70362)
[![IET Image Processing](https://img.shields.io/badge/Published%20in-IET%20Image%20Processing-orange)](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70362)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the models presented in:

> **Towards Lightweight Image Recognition: Multi-Column Convolutional Neural Networks Based on Learned Separable Transform**
> Maxim Vashkevich, Egor Krivalcevich
> *IET Image Processing*, 2026
> [DOI: 10.1049/ipr2.70362](https://doi.org/10.1049/ipr2.70362)

---

## Abstract

This paper introduces two novel neural network architectures based on the **learned separable transform (LST)** for efficient image recognition.

First, we present **MCNN-LST**, a hybrid architecture that integrates multi-column convolutional networks with LST blocks to compress multi-scale features into compact, discriminative embeddings. Second, we propose **MCNN-Eff-LST**, which employs mobile inverted bottleneck convolutions to extract efficient multi-scale representations and uses LST as a learnable alternative to global average pooling (GAP).

| Model | Dataset | Accuracy | Parameters | FLOPs |
|---|---|---|---|---|
| **MCNN-LST** | Fashion-MNIST | **93.69 %** | 57.7 K | 1.8 M |
| **MCNN-Eff-LST** | CIFAR-10 | **84.40 %** | 37.7 K | 5.5 M |

Ablation studies confirm the contribution of LST blocks: removing them causes accuracy drops of **1.69 %** and **2.74 %** for MCNN-LST and MCNN-Eff-LST, respectively. LST-based models offer a strong trade-off between parameter efficiency and recognition performance, making them well suited for **edge-computing applications**.

---

## Repository structure

```
MCNN-LST-Image-Recognition/
├── l2dst_lib/                     # Core library: Learned 2D Separable Transform
│   ├── __init__.py
│   └── lst_nn.py                  # L2DST, LST_1, multichan_to_2D
├── EfficientNet_lib.py            # MBConvBlock used by MCNN-Eff-LST
│
├── MCNN-LST-FashionMNIST.ipynb    # MCNN-LST: full study notebook (Fashion-MNIST)
├── MCNN-Eff-LST-CIFAR10.ipynb     # MCNN-Eff-LST: full study notebook (CIFAR-10)
│
├── pretrain/                      # Pretrained model checkpoints
│   ├── MCNN_Eff_LST24_ker16_mbCh36.pth
│   └── MCNN_LST_28_ker36.pth
│
├── requirements.txt
├── LICENSE
└── README.md
```

Each notebook is split into eight self-contained sections that share the same layout:

| § | Section | What it does |
|---|---|---|
| 1 | Imports & device | PyTorch / utility imports, picks CUDA if available |
| 2 | Dataset | Loads Fashion-MNIST or CIFAR-10 (auto-download) |
| 3 | Model | Defines `MultiConv4_LST` / `EffNet_LST` |
| 4 | Helpers | `acc_estimate`, `train_test_loop` |
| 5 | **Pretrained → test** | Loads the released checkpoint and reports test accuracy |
| 6 | **Single training run** | Trains one model with user-tunable hyperparameters and saves the best checkpoint |
| 7 | **Optuna search** | Runs/resumes the hyperparameter search reported in the paper |
| 8 | **10-run statistics** | Trains the best configuration ten times, reports mean ± std |

Sections 5–8 are independent: once §1–§4 are executed you can jump directly to whichever section you need.

---

## Model architectures

### MCNN-LST (Fashion-MNIST, 1 × 28 × 28)

Four parallel convolutional branches extract features at different scales; each is followed by an LST block that compresses spatial information into a compact embedding.

```
Input (1×28×28)
  ├── Conv2D 2×2 ──► MaxPool 2×2 ──► multichan_to_2D ──► L2DST
  ├── Conv2D 3×3 ──► MaxPool 2×2 ──► multichan_to_2D ──► L2DST
  ├── Conv2D 4×4 ──► MaxPool 2×2 ──► multichan_to_2D ──► L2DST
  └── Conv2D 5×5 ──► MaxPool 2×2 ──► multichan_to_2D ──► L2DST
        │
   Concat ──► BatchNorm ──► Dropout ──► FC (10 classes)
```

**Parameters:** 57.7 K **FLOPs:** 1.8 M **Accuracy:** 93.69 %

### MCNN-Eff-LST (CIFAR-10, 3 × 32 × 32)

Extends MCNN-LST by inserting EfficientNet-style MBConv blocks between the multi-scale stem and the LST, replacing global average pooling with a learnable alternative.

```
Input (3×32×32)
  ├── Conv2D 2×2 ──► MaxPool 2×2
  ├── Conv2D 3×3 ──► MaxPool 2×2
  ├── Conv2D 4×4 ──► MaxPool 2×2
  └── Conv2D 5×5 ──► MaxPool 2×2
        │
   Concat ──► MBConv₁ (e=1) ──► MBConv₂ (e=6) ──► MBConv₃ (e=6)
        │
   multichan_to_2D ──► L2DST ──► BatchNorm ──► Dropout ──► FC (10 classes)
```

**Parameters:** 37.7 K **FLOPs:** 5.5 M **Accuracy:** 84.40 %

---

## Ablation studies

| Variant | Accuracy | Drop vs. full model |
|---|---|---|
| **MCNN-LST** (full) | **93.69 %** | — |
| MCNN-LST *w/o* LST | 92.00 % | −1.69 % |
| **MCNN-Eff-LST** (full) | **84.40 %** | — |
| MCNN-Eff-LST *w/o* LST | 81.66 % | −2.74 % |

Removing the LST block and replacing it with global average pooling consistently degrades accuracy, confirming that the learned separable transform provides meaningful spatial feature aggregation beyond simple averaging.

---

## Hyperparameter search space (Optuna)

| Parameter | Symbol | Range / values | Sampling |
|---|---|---|---|
| Initial learning rate | η<sub>max</sub> | [1·10⁻⁴, 3·10⁻²] | Log-uniform |
| Weight decay | λ | [1·10⁻⁸, 1·10⁻⁴] (Fashion) / [5·10⁻¹², 1·10⁻⁸] (CIFAR) | Log-uniform |
| FC dropout | p<sub>drop_fc</sub> | [0.01, 0.55]  | Uniform |
| LST dropout | p<sub>drop_lst</sub> | [0.01, 0.55]  | Uniform |
| Annealing loops | T₀ | {20, 10, 4, 2}  | Categorical |
| Batch size | — | 1024 | Fixed |
| Training epochs | — | 200  | Fixed |

The learning rate follows a **Cosine Annealing with Warm Restarts** schedule with η<sub>min</sub> = 0.01 · η<sub>max</sub>.

---

## Getting started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- A CUDA-capable GPU is recommended (training will fall back to CPU automatically)

### Installation
git@github.com:Mak-Sim/MCNN-LST-Image-Recognition.git
```bash
git clone git@github.com:Mak-Sim/MCNN-LST-Image-Recognition.git
cd MCNN-LST-Image-Recognition
pip install -r requirements.txt
```

### Usage

Both notebooks share an identical eight-section layout described above.
Open the one that corresponds to your dataset of interest:

- [`MCNN-LST-FashionMNIST.ipynb`](MCNN-LST-FashionMNIST.ipynb)
- [`MCNN-Eff-LST-CIFAR10.ipynb`](MCNN-Eff-LST-CIFAR10.ipynb)

Recommended workflow:

1. Run **§1–§4** (imports, dataset, model, helpers).
2. Run **§5** to load the released checkpoint and reproduce the reported test accuracy.
3. Edit the constants at the top of **§6** to train a single model with your own hyperparameters; the best checkpoint is saved under `model_backup/{Dataset}/<MODEL_NAME>/`.
4. Run **§7** to launch (or resume) the Optuna search. Trials are persisted in a local SQLite file (`db_fashion.sqlite3` / `db_cifar.sqlite3`); the CIFAR-10 study used in the paper is shipped with this repository.
5. Run **§8** to evaluate the best configuration over 10 independent runs and write a results table to `results_*.txt`.

Training progress is logged to TensorBoard under `runs/`:

```bash
tensorboard --logdir runs
```

---

## Key components

### LST (Learned 2D Separable Transform)

The core building block of both architectures. Given a 2-D input (e.g. a feature map reshaped from channels), L2DST applies:

1. **Row-wise** linear transform with a `tanh` (or `gelu`) activation
2. **Column-wise** linear transform with a `tanh` (or `gelu`) activation
3. **Output**: a compact 2-D latent representation

This separable design reduces parameter count compared to a full 2-D linear layer while retaining the ability to model global spatial dependencies.

| Class | Activation | Role |
|---|---|---|
| `L2DST` | tanh | Baseline LST used in MCNN-LST and MCNN-Eff-LST |
| `L2DST_ge` | GELU | Variant explored in the ablation study |
| `LST_1` | tanh | Alternative LST block |
| `multichan_to_2D` | — | Reshapes (B, C, H, W) → (B, s·H, s·W) where s = √C |

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{vashkevich_mcnn_lst_2026,
  author    = {Vashkevich, Maxim and Krivalcevich, Egor},
  title     = {Towards Lightweight Image Recognition: Multi-Column Convolutional
               Neural Networks Based on Learned Separable Transform},
  journal   = {IET Image Processing},
  year      = {2026},
  publisher = {Wiley},
  doi       = {10.1049/ipr2.70362}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

For questions, issues, or collaboration opportunities, please open a GitHub issue or contact the corresponding author of the paper.
