# Dataset Setup

Raw chest X-ray images are not stored in this repository because of their size, licensing conditions, and medical-data considerations.

## Primary dataset

Use the public **Chest X-Ray (Pneumonia, COVID-19, Tuberculosis)** collection referenced in the project report. It contains four classes and 7,135 images across predefined training, validation, and test splits.

Expected directory structure:

```text
data/raw/primary/
├── train/
│   ├── COVID19/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TURBERCULOSIS/
├── val/
│   ├── COVID19/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TURBERCULOSIS/
└── test/
    ├── COVID19/
    ├── NORMAL/
    ├── PNEUMONIA/
    └── TURBERCULOSIS/
```

Some dataset distributions use slightly different folder spellings. Update the notebook configuration without changing the class order used by the trained checkpoints.

## External calibration dataset

The secondary CXR collection is used only for ensemble-weight calibration. Before calibration, compare it against every image in the primary train, validation, and test splits using:

1. SHA-256 exact matching
2. Full-image and center-crop perceptual hashing
3. SSIM verification of candidate pairs
4. Exclusion of exact, strong, and possible overlaps

The reported strict manifest retains 52,927 images. Do not use the original test labels to optimize ensemble weights.

## Reproducibility checks

- Confirm every image is readable.
- Print per-class counts for each split.
- Verify the class-to-index mapping for every model.
- Apply random augmentation only to the training split.
- Keep validation and test transforms deterministic.
- Record the exact dataset source, version, and download date.
