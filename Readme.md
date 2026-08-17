📦 Datasets & Pre-trained Models1. RNH6K Dataset DownloadDownload the synthesized benchmark dataset via Google Drive:RNH6K Benchmark (RNH6K-H & RNH6K-NH): Google Drive Download LinkAfter downloading, unzip and place the folders directly under the data/ directory:PlaintextPGERNet/
└── data/
    ├── RNH6K-H/
    │   ├── train/ (hazy, vis, nir)
    │   ├── val/
    │   └── test/
    └── RNH6K-NH/
        ├── train/ (hazy, vis, nir)
        ├── val/
        └── test/
2. Pre-trained Model CheckpointsWe provide two pre-trained checkpoints trained on different haze subsets. They can be downloaded here:Model VariantTraining SubsetParams (M)FLOPs (G)Test PSNR (dB)Download LinkPGERNet-BRNH6K-NH6.4724.7222.54Google DrivePGERNet-LRNH6K-H103.45392.2824.23Google Drive(Note: The quantitative results above reflect the performance on their respective testing subsets as reported in the paper.)  Download the .pth files and place them into the trained_models/ directory (e.g., trained_models/PGERNet_val.pth or trained_models/PGERNet.pth).⚙️ InstallationPrerequisitesPython 3.8+PyTorch (tested on 2.0+)CUDA-enabled GPU (e.g., RTX 3090)DependenciesInstall the required packages using pip:Bashpip install torch torchvision numpy Pillow scikit-image kornia thop
🚀 Usage1. TrainingThe training parameters, dataset paths, and loss configurations (utilizing the CompositeDehazingLoss combining L1, Edge, and FFT losses) are managed within the Config class in main.py.To train the model from scratch, run:Bashpython main.py
Note: The script will automatically save checkpoints to the trained_models/ directory and log metrics to log/PGERNet.txt. It executes training and subsequently runs testing on the best validation model.2. Testing / EvaluationTo evaluate using downloaded or pre-trained checkpoints:Place the downloaded checkpoint in the trained_models/ directory (e.g., trained_models/PGERNet_val.pth).Comment out the train(cfg, model) line in the if __name__ == "__main__": block in main.py.Run the evaluation script to calculate PSNR and SSIM on the test split:Bashpython main.py
3. Model Complexity (FLOPs & Params)To profile the computational complexity (FLOPs, parameter count, and inference time) for the scalable variants, run:Bashpython network/PGERNet.py
📐 Model VariantsPGERNet provides scalable architectural variants controlled by the base dimension (dim) and encoder/decoder block depths:  PGERNet-B (Base): Compact model (~6.47M parameters, 24.72G FLOPs), optimal efficiency-accuracy trade-off.  PGERNet-M (Medium): Increased block depth ([2,2,2,2]) for enhanced structural representation.  PGERNet-L (Large): High-capacity model (dim=64, 103.45M parameters) establishing state-of-the-art restoration performance.
