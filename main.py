import os
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import torch
from torch import nn, einsum
import torch.nn.functional as F
from kornia.losses import SSIMLoss
from network.PGERNet import PGERNet

netsource = 'PGERNet'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    lr = 1e-4
    final_lr = 6e-5
    warmup_epochs = 10
    epochs = 50
    img_size = 256
    dim = 16
    seed = 66
    dataset_name = "RNH6K-NH"
    train_hazy = f"data/{dataset_name}/train/hazy"
    train_clean = f"data/{dataset_name}/train/vis"
    train_nir = f"data/{dataset_name}/train/nir"
    val_hazy = f"data/{dataset_name}/val/hazy"
    val_clean = f"data/{dataset_name}/val/vis"
    val_nir = f"data/{dataset_name}/val/nir"
    test_hazy = f"data/{dataset_name}/test/hazy"
    test_clean = f"data/{dataset_name}/test/vis"
    test_nir = f"data/{dataset_name}/test/nir"
    save_dir = "results"
    ablation_log = "log/ablation.txt"
    model_base_dir = "trained_models"
    model_filename = netsource + ".pth"
    train_log = "log/" + netsource + ".txt"

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(k, requires_grad=False)
        self.weight_y = nn.Parameter(k.transpose(2, 3), requires_grad=False)

    def forward(self, x, y):
        c = x.size(1)
        wx = self.weight_x.repeat(c, 1, 1, 1).to(x.device)
        wy = self.weight_y.repeat(c, 1, 1, 1).to(x.device)
        gx_x = F.conv2d(x, wx, padding=1, groups=c)
        gy_x = F.conv2d(x, wy, padding=1, groups=c)
        gx_y = F.conv2d(y, wx, padding=1, groups=c)
        gy_y = F.conv2d(y, wy, padding=1, groups=c)
        return torch.mean(torch.abs(gx_x - gx_y)) + torch.mean(torch.abs(gy_x - gy_y))

class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, x, y):
        x_f32 = x.to(torch.float32)
        y_f32 = y.to(torch.float32)
        fft_x = torch.fft.rfft2(x_f32)
        fft_y = torch.fft.rfft2(y_f32)
        return torch.mean(torch.abs(fft_x - fft_y))

class CompositeDehazingLoss(nn.Module):
    def __init__(self, device):
        super(CompositeDehazingLoss, self).__init__()
        self.l1 = nn.L1Loss().to(device)
        self.edge = EdgeLoss().to(device)
        self.fft = FFTLoss().to(device)

    def forward(self, pred, clean, pred_01, clean_01):
        l_1 = self.l1(pred, clean)
        l_edge = self.edge(pred_01, clean_01)
        l_fft = self.fft(pred_01, clean_01)
        total_loss = l_1 + 0.05 * l_edge + 0.01 * l_fft
        return total_loss, l_1, l_edge, l_fft

class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, clean_dir, nir_dir, is_train=True):
        self.is_train = is_train
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir
        self.nir_dir = nir_dir
        self.hazy_filenames = [f for f in os.listdir(hazy_dir) if f.endswith(('.png', '.tiff'))]
        self.clean_nir_filenames = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.tiff'))]
        self.transform_rgb = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_nir = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.hazy_filenames)

    def __getitem__(self, idx):
        hazy_filename = self.hazy_filenames[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_filename)
        clean_path = os.path.join(self.clean_dir, hazy_filename)
        nir_path = os.path.join(self.nir_dir, hazy_filename)
        hazy = Image.open(hazy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')
        nir = Image.open(nir_path).convert('L')
        if self.is_train:
            if random.random() > 0.5:
                hazy = hazy.transpose(Image.FLIP_LEFT_RIGHT)
                clean = clean.transpose(Image.FLIP_LEFT_RIGHT)
                nir = nir.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hazy = hazy.transpose(Image.FLIP_TOP_BOTTOM)
                clean = clean.transpose(Image.FLIP_TOP_BOTTOM)
                nir = nir.transpose(Image.FLIP_TOP_BOTTOM)
        return (
            self.transform_rgb(hazy),
            self.transform_rgb(clean),
            self.transform_nir(nir)
        )

def validate(cfg, model):
    val_set = DehazeDataset(cfg.val_hazy, cfg.val_clean, cfg.val_nir, is_train=False)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    val_loader = DataLoader(val_set, batch_size=2 * cfg.batch_size, shuffle=False,
                            num_workers=4, worker_init_fn=seed_worker, generator=g)
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for hazy, clean, nir in val_loader:
            hazy = hazy.to(cfg.device)
            clean = clean.to(cfg.device)
            nir = nir.to(cfg.device)
            with torch.amp.autocast(cfg.device):
                output = model(hazy, nir)
            output = (output.clamp(-1, 1) + 1) / 2.0 * 255.0
            clean = (clean + 1) / 2.0 * 255.0
            output_np = output.cpu().numpy().transpose(0, 2, 3, 1)
            clean_np = clean.cpu().numpy().transpose(0, 2, 3, 1)
            for i in range(output.size(0)):
                pred = output_np[i].astype(np.uint8)
                gt = clean_np[i].astype(np.uint8)
                current_psnr = psnr(gt, pred, data_range=255)
                current_ssim = ssim(gt, pred, data_range=255, channel_axis=2, multichannel=True)
                total_psnr += current_psnr
                total_ssim += current_ssim
    avg_psnr = total_psnr / len(val_set)
    avg_ssim = total_ssim / len(val_set)
    model.train()
    return avg_psnr, avg_ssim

def train(cfg, model):
    model_path = os.path.join(cfg.model_base_dir, cfg.model_filename)
    os.makedirs(cfg.model_base_dir, exist_ok=True)
    os.makedirs("log", exist_ok=True)
    train_set = DehazeDataset(cfg.train_hazy, cfg.train_clean, cfg.train_nir)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, worker_init_fn=seed_worker, generator=g)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = CompositeDehazingLoss(cfg.device)
    scaler = torch.amp.GradScaler(cfg.device)
    best_loss = float('inf')
    best_val_psnr = 0.0
    for epoch in range(cfg.epochs):
        start_time = time.time()
        if epoch < cfg.warmup_epochs:
            lr = (epoch + 1) / cfg.warmup_epochs * cfg.lr
        else:
            progress = (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)
            lr = cfg.lr - (cfg.lr - cfg.final_lr) * progress
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        model.train()
        epoch_loss = 0.0
        for i, (hazy, clean, nir) in enumerate(train_loader):
            hazy = hazy.to(cfg.device)
            clean = clean.to(cfg.device)
            nir = nir.to(cfg.device)
            opt.zero_grad()
            with torch.amp.autocast(cfg.device):
                pred, aux_preds = model(hazy, nir)
                pred_01 = (pred + 1) / 2.0
                clean_01 = (clean + 1) / 2.0
                main_loss, l_1_main, l_edge_main, l_fft_main = criterion(pred, clean, pred_01, clean_01)
                aux_loss = 0.0
                for aux in aux_preds:
                    clean_down = F.interpolate(clean, size=aux.shape[2:], mode='bilinear', align_corners=False)
                    aux_loss += F.l1_loss(aux, clean_down)
                total_loss = main_loss + 0.2 * aux_loss
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            epoch_loss += total_loss.item()
            if i % 200 == 0:
                log_info = (f"Epoch [{epoch + 1}/{cfg.epochs}] Batch [{i}/{len(train_loader)}] "
                            f"Total Loss: {total_loss.item():.4f} (Main: {main_loss.item():.4f} "
                            f"[L1: {l_1_main.item():.4f}, Edge: {l_edge_main.item():.4f}, FFT: {l_fft_main.item():.4f}], "
                            f"Aux: {aux_loss.item():.4f}) LR: {lr:.2e}")
                print(log_info)
                with open(cfg.train_log, "a") as f:
                    f.write(log_info + "\n")
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        minutes, seconds = divmod(epoch_time, 60)
        time_str = f"{int(minutes)}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"
        log_info = f"Epoch [{epoch + 1}/{cfg.epochs}] Avg Loss: {avg_loss:.4f} LR: {lr:.2e} Time: {time_str}"
        print(log_info)
        with open(cfg.train_log, "a") as f:
            f.write(log_info + "\n")
        if (epoch + 1) % 2 == 0:
            val_psnr, val_ssim = validate(cfg, model)
            val_log = f"----------Validation @ Epoch {epoch + 1} - PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}----------"
            print(val_log)
            with open(cfg.train_log, "a") as f:
                f.write(val_log + "\n")
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                save_path_val = model_path.replace(".pth", "_val.pth")
                torch.save(model.state_dict(), save_path_val)
                sv_log = f"Saved best validation model to {save_path_val} with PSNR: {best_val_psnr:.2f}"
                print(sv_log)
                with open(cfg.train_log, "a") as f:
                    f.write(sv_log + "\n")
        if avg_loss < best_loss:
            best_loss = avg_loss

def test(cfg, model):
    val_name = cfg.model_filename
    model_path_to_load = os.path.join(cfg.model_base_dir, val_name.replace(".pth", "_val.pth"))
    if not os.path.exists(model_path_to_load):
        print(f"Error: Model not found at {model_path_to_load}. Please train the model first.")
        return
    model.load_state_dict(torch.load(model_path_to_load, map_location=cfg.device, weights_only=False))
    model.eval()
    test_set = DehazeDataset(cfg.test_hazy, cfg.test_clean, cfg.test_nir, is_train=False)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False,
                             num_workers=5, worker_init_fn=seed_worker, generator=g)
    os.makedirs(cfg.save_dir, exist_ok=True)
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for batch_idx, (hazy, clean, nir) in enumerate(test_loader):
            hazy = hazy.to(cfg.device)
            clean = clean.to(cfg.device)
            nir = nir.to(cfg.device)
            with torch.amp.autocast(cfg.device):
                output = model(hazy, nir)
            output = (output.clamp(-1, 1) + 1) / 2.0 * 255.0
            clean = (clean + 1) / 2.0 * 255.0
            output_np = output.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
            clean_np = clean.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
            for i in range(output.size(0)):
                fn = test_set.hazy_filenames[batch_idx * test_loader.batch_size + i]
                pred = output_np[i]
                gt = clean_np[i]
                current_psnr = psnr(gt, pred, data_range=255)
                current_ssim = ssim(gt, pred, data_range=255, channel_axis=2, multichannel=True)
                total_psnr += current_psnr
                total_ssim += current_ssim
    overall_avg_psnr = total_psnr / len(test_set)
    overall_avg_ssim = total_ssim / len(test_set)
    log_info = f"{netsource} Overall PSNR: {overall_avg_psnr:.2f} dB, Overall SSIM: {overall_avg_ssim:.4f}"
    print(log_info)
    with open(cfg.ablation_log, "a") as f:
        f.write(log_info + "\n")

if __name__ == "__main__":
    cfg = Config()
    set_seed(cfg.seed)
    model = PGERNet(dim=cfg.dim, dim_mults=(1, 2, 4, 5),
                    num_blocks_encoder=[1, 1, 1, 1],
                    num_blocks_decoder=[1, 1, 1, 1]).to(cfg.device)
    print("Starting training...")
    #train(cfg, model)
    print("Training finished. Starting testing...")
    test(cfg, model)
    print("Testing finished.")
