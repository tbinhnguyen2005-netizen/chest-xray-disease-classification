# Deep Learning and Ensemble Learning for Chest X-Ray Classification

An academic computer vision project for classifying chest X-ray images into four categories: **COVID-19, Normal, Pneumonia, and Tuberculosis**. The study compares convolutional and transformer architectures, evaluates Coordinate Attention (CA), and develops a leakage-controlled weighted soft-voting ensemble.

> This repository documents an academic experiment and is not intended for clinical diagnosis or medical decision-making.

## Highlights

- Evaluated EfficientNet variants B0-B7, DenseNet121, and Swin-Tiny using ImageNet transfer learning.
- Improved EfficientNet-B1 with Coordinate Attention from **92.48% to 94.42% test accuracy** and from **92.92% to 95.97% macro F1-score**.
- Removed overlap from an external calibration dataset using SHA-256, perceptual hashing, and SSIM before ensemble-weight optimization.
- Optimized soft-voting weights with SLSQP, achieving the lowest final test log-loss of **0.1605**.

## Dataset

The primary public dataset contains **7,135 chest X-ray images**.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| COVID-19 | 460 | 10 | 106 | 576 |
| Normal | 1,341 | 8 | 234 | 1,583 |
| Pneumonia | 3,875 | 8 | 390 | 4,273 |
| Tuberculosis | 650 | 12 | 41 | 703 |
| **Total** | **6,326** | **38** | **771** | **7,135** |

The original validation set is very small, so a secondary four-class CXR collection was used only to calibrate ensemble weights. Duplicate filtering reduced 72,297 readable external images to **52,927 strict calibration images**.

Dataset download instructions and the required folder layout are documented in [`data/README.md`](data/README.md). Raw medical images are not redistributed in this repository.

## Methodology

### Preprocessing and training

- Model-specific input resizing
- ImageNet normalization
- Training-only augmentation: horizontal flipping, rotation, brightness/contrast jitter, and random resized cropping where applicable
- Weighted sampling and class-weighted cross-entropy for class imbalance
- AdamW optimization, learning-rate scheduling, early stopping, and mixed-precision training
- Best-checkpoint selection using validation performance

### Models

1. **EfficientNet B0-B7 screening** to study the accuracy-compute trade-off.
2. **EfficientNet-B1 attention ablation** with CA, CBAM, and combined CA-CBAM at different insertion points.
3. **DenseNet121 and DenseNet121+CA** comparison.
4. **Swin-Tiny** as a hierarchical transformer baseline.
5. **Weighted soft voting** across EfficientNet-B1+CA, Swin-Tiny, and DenseNet121+CA.

Coordinate Attention was most effective when inserted after the final EfficientNet-B1 feature block and before global average pooling.

## Main Results

### EfficientNet-B1 attention ablation

| Model | Parameters | Accuracy | Macro F1 |
|---|---:|---:|---:|
| EfficientNet-B1 baseline | 6.52M | 92.48% | 92.92% |
| **EfficientNet-B1+CA (last)** | **6.67M** | **94.42%** | **95.97%** |

CA improved accuracy by **1.94 percentage points** and macro F1 by **3.05 percentage points**, while adding approximately **0.15 million parameters**.

### Final comparison on the untouched 771-image test set

| Method | Accuracy | Macro F1 | Weighted F1 | Log-loss |
|---|---:|---:|---:|---:|
| **EfficientNet-B1+CA** | **94.42%** | **95.97%** | **94.41%** | 0.1645 |
| Swin-Tiny | 93.51% | 93.49% | 93.54% | 0.1751 |
| DenseNet121+CA | 92.35% | 92.11% | 92.31% | 0.3369 |
| Equal-weight ensemble | 93.64% | 94.24% | 93.63% | 0.1670 |
| Optimized ensemble | 94.03% | 94.51% | 94.03% | **0.1605** |

EfficientNet-B1+CA remained the strongest method by accuracy and macro F1. The optimized ensemble did not surpass it on those metrics, but produced the best probability quality as measured by log-loss.

## Ensemble Calibration

The ensemble probability is:

```text
P = 0.2516 * P_EfficientNet + 0.6847 * P_Swin + 0.0637 * P_DenseNet
```

Weights were constrained to be non-negative and sum to one. They were selected by minimizing class-balanced log-loss with SLSQP on the strict external calibration set. The original 771-image test labels were not used to select these weights.

## Evaluation

The experiments report:

- Accuracy
- Macro precision, recall, and F1-score
- Weighted F1-score
- One-vs-rest AUC
- Multiclass log-loss
- Confusion matrices and per-class classification reports
- Grad-CAM visualizations for selected EfficientNet-B1 variants

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/        # Experiment notebooks (added separately)
├── assets/           # Figures, diagrams, and selected results
└── reports/          # Final academic report
```

Large datasets, cached predictions, ZIP archives, and trained checkpoints are intentionally excluded from Git history.

## Environment

The experiments were developed with Python and PyTorch and executed primarily on Kaggle/Google Colab with an NVIDIA Tesla T4 GPU.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Exact package versions should be recorded from the experiment environment when rerunning the notebooks.

## Team

- Nguyen Thanh Binh
- Tran Ba Le Hoang
- Nguyen Thi Ngoc Van
- Pham Tuan Thanh

This was a team research project. Individual responsibilities should be documented clearly before using the repository as evidence of personal contribution.

## Limitations

- The original validation set contains only 38 images.
- The final evaluation uses one predefined test split and does not report confidence intervals or cross-validation.
- The external calibration data come from a different distribution despite strict duplicate filtering.
- The models have not been validated on an independent clinical site or at patient level.
- Results should not be interpreted as clinical diagnostic performance.

## Future Work

- Patient-level and multi-site validation
- Cross-validation and statistical confidence intervals
- Probability calibration and uncertainty estimation
- Lung-region localization and additional explainability analysis
- Evaluation under acquisition and domain shifts

## Acknowledgment

The project uses publicly available chest X-ray datasets and ImageNet-pretrained model weights. Please review the original dataset licenses and citation requirements before redistributing any data or derived artifacts.
