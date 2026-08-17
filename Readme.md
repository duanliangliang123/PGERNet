## 📦 Datasets & Pre-trained Models

### 1. RNH6K Dataset Download
Download the synthesized benchmark dataset via Google Drive:
- **RNH6K Benchmark (RNH6K-H & RNH6K-NH):** [Google Drive Download Link](#) *(Add your actual link here)*

After downloading, unzip and place the folders directly under the `data/` directory:

```text
PGERNet/
└── data/
    ├── RNH6K-H/
    │   ├── train/ (hazy, vis, nir)
    │   ├── val/
    │   └── test/
    └── RNH6K-NH/
        ├── train/ (hazy, vis, nir)
        ├── val/
        └── test/

2. Pre-trained Model CheckpointsWe provide two pre-trained checkpoints trained on different haze subsets. They can be downloaded here:Model VariantTraining SubsetParams (M)FLOPs (G)Test PSNR (dB)Download LinkPGERNet-BRNH6K-NH6.4724.7222.54Google DrivePGERNet-LRNH6K-H103.45392.2824.23Google DriveNote: The quantitative results above reflect the performance on their respective testing subsets as reported in the paper.Download the .pth files and place them into the trained_models/ directory (e.g., trained_models/PGERNet_val.pth or trained_models/PGERNet.pth).
