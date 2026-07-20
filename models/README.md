# Model Checkpoints

Trained model checkpoints are intentionally excluded from Git history.

Expected final checkpoints:

```text
models/checkpoints/
├── best_EfficientNet_B1_CA_Last.pth
├── best_Swin_Tiny.pth
└── densenet121_ca_best.pth
```

If checkpoints are shared, publish them through an appropriate release or model-hosting service and document:

- File name and download URL
- SHA-256 checksum
- Model architecture and number of classes
- Class order used during training
- Input resolution and normalization
- Dataset version and split
- Validation criterion used to select the checkpoint

Do not commit API keys, private storage URLs, or unrestricted write credentials with checkpoint metadata.
