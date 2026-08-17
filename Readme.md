PGERNet
📦 Datasets & Pre-trained Models
RNH6K Dataset Download
Download the synthesized benchmark dataset via Google Drive:
RNH6K Benchmark (RNH6K-H & RNH6K-NH): [Google Drive Download Link](Add your actual link here)
After downloading, unzip and place the folders directly under the data/ directory:
plain
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
Pre-trained Model Checkpoints
We provide two pre-trained checkpoints trained on different haze subsets:
表格
Model	Training Subset	Params	FLOPs	Test PSNR	Download
PGERNet-B	RNH6K-NH	6.47 M	24.72 G	22.54 dB	[Google Drive](Add your actual link here)
PGERNet-L	RNH6K-H	103.45 M	392.28 G	24.23 dB	[Google Drive](Add your actual link here)
Note: The quantitative results above reflect the performance on their respective testing subsets as reported in the paper.
Download the .pth files and place them into the trained_models/ directory (e.g., trained_models/PGERNet_val.pth or trained_models/PGERNet.pth).
⚙️ Installation
Prerequisites
Python 3.8+
PyTorch (tested on 2.0+)
CUDA-enabled GPU (e.g., RTX 3090)
Dependencies
Install the required packages using pip:
bash
pip install torch torchvision numpy Pillow scikit-image kornia thop
🚀 Usage
Training
The training parameters, dataset paths, and loss configurations (utilizing the CompositeDehazingLoss combining L1, Edge, and FFT losses) are managed within the Config class in main.py.
To train the model from scratch, run:
bash
python main.py
Note: The script will automatically save checkpoints to the trained_models/ directory and log metrics to log/PGERNet.txt. It executes training and subsequently runs testing on the best validation model.
Testing / Evaluation
To evaluate using downloaded or pre-trained checkpoints:
Place the downloaded checkpoint in the trained_models/ directory (e.g., trained_models/PGERNet_val.pth).
Comment out the train(cfg, model) line in the if __name__ == "__main__": block in main.py.
Run the evaluation script to calculate PSNR and SSIM on the test split:
bash
python main.py
Model Complexity (FLOPs & Params)
To profile the computational complexity (FLOPs, parameter count, and inference time) for the scalable variants, run:
bash
python network/PGERNet.py
📐 Model Variants
PGERNet provides scalable architectural variants controlled by the base dimension (dim) and encoder/decoder block depths:
表格
Variant	Description
PGERNet-B (Base)	Compact model (~6.47M parameters, 24.72G FLOPs), optimal efficiency-accuracy trade-off.
PGERNet-M (Medium)	Increased block depth ([2,2,2,2]) for enhanced structural representation.
PGERNet-L (Large)	High-capacity model (dim=64, 103.45M parameters) establishing state-of-the-art restoration performance.
