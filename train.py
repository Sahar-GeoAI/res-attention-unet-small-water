import os
import glob
import json
import time
import random
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

TRAIN_IMAGE_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"
VAL_IMAGE_DIR = "data/val/images"
VAL_MASK_DIR = "data/val/masks"
OUTPUT_DIR = "output"

IN_CHANNELS = 11
NUM_CLASSES = 1
INIT_FEATURES = 32
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 5
ETA_MIN = 1e-6
GRAD_CLIP = 0.5
SWA_EPOCHS = 20
SWA_LR = 1e-4
SWA_ANNEAL_EPOCHS = 5
BCE_WEIGHT = 0.3
DICE_WEIGHT = 0.4
FOCAL_WEIGHT = 0.3
FOCAL_ALPHA = 0.7
FOCAL_GAMMA = 2.0
POS_WEIGHT = 2.0
DS_WEIGHTS = (0.70, 0.15, 0.15)
DROPOUT_BOTTLENECK = 0.30
DROPOUT_DEC4 = 0.20
DROPOUT_DEC3 = 0.20
DROPOUT_DEC2 = 0.15
AUG_HFLIP = 0.5
AUG_VFLIP = 0.5
AUG_ROT90 = 0.5
AUG_AFFINE_P = 0.5
AUG_MAX_ROT_DEG = 30.0
AUG_SCALE_RANGE = (0.9, 1.1)
AUG_ELASTIC_P = 0.3
ELASTIC_ALPHA = 40.0
ELASTIC_SIGMA = 6.0
WATER_OVERSAMPLE = 1.5
NUM_WORKERS = 0
SEEDS = [42, 123, 456, 789]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class GeometricAugmentation:
    def __init__(self):
        self.hflip_p = AUG_HFLIP
        self.vflip_p = AUG_VFLIP
        self.rot90_p = AUG_ROT90
        self.affine_p = AUG_AFFINE_P
        self.max_rot_deg = AUG_MAX_ROT_DEG
        self.scale_range = AUG_SCALE_RANGE
        self.elastic_p = AUG_ELASTIC_P
        self.elastic_alpha = ELASTIC_ALPHA
        self.elastic_sigma = ELASTIC_SIGMA

    def _affine(self, image, mask):
        c, h, w = image.shape
        angle = random.uniform(-self.max_rot_deg, self.max_rot_deg)
        scale = random.uniform(*self.scale_range)
        theta = np.deg2rad(angle)
        cos, sin = np.cos(theta), np.sin(theta)
        m = np.array([[cos, sin], [-sin, cos]], dtype=np.float64) / scale
        center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
        offset = center - m @ center
        out_img = np.empty_like(image)
        for i in range(c):
            out_img[i] = ndi.affine_transform(image[i], m, offset=offset, order=1, mode="reflect")
        out_mask = ndi.affine_transform(mask, m, offset=offset, order=0, mode="reflect")
        return out_img, out_mask

    def _elastic(self, image, mask):
        c, h, w = image.shape
        dy = ndi.gaussian_filter(np.random.rand(h, w) * 2 - 1, self.elastic_sigma, mode="reflect") * self.elastic_alpha
        dx = ndi.gaussian_filter(np.random.rand(h, w) * 2 - 1, self.elastic_sigma, mode="reflect") * self.elastic_alpha
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = np.array([yy + dy, xx + dx])
        out_img = np.empty_like(image)
        for i in range(c):
            out_img[i] = ndi.map_coordinates(image[i], coords, order=1, mode="reflect")
        out_mask = ndi.map_coordinates(mask, coords, order=0, mode="reflect")
        return out_img, out_mask

    def __call__(self, image, mask):
        if random.random() < self.hflip_p:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < self.vflip_p:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < self.rot90_p:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
        if random.random() < self.affine_p:
            image, mask = self._affine(image, mask)
        if random.random() < self.elastic_p:
            image, mask = self._elastic(image, mask)
        mask = (mask > 0.5).astype(np.float32)
        return (np.ascontiguousarray(image, dtype=np.float32),
                np.ascontiguousarray(mask, dtype=np.float32))


def list_tifs(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(folder, "*.tiff")))
    return paths


def mask_has_water(path):
    with rasterio.open(path) as src:
        return float(src.read(1).sum()) > 0.0


class WaterDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_paths = list_tifs(image_dir)
        self.mask_paths = list_tifs(mask_dir)
        if not self.image_paths or len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Image and mask files do not match in {image_dir} and {mask_dir}")
        self.transform = GeometricAugmentation() if augment else None
        with ThreadPoolExecutor(max_workers=8) as executor:
            self.has_water = list(executor.map(mask_has_water, self.mask_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        image_t = torch.from_numpy(image.copy()).float()
        mask_t = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        return image_t, mask_t, self.has_water[idx]


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.shortcut(x))


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_gate(g)
        x1 = self.W_skip(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(g1 + x1))
        return x * alpha


class ResidualAttentionUNet(nn.Module):
    def __init__(self, in_channels=11, num_classes=1, init_features=32,
                 dropout_bottleneck=0.30, dropout_dec4=0.20, dropout_dec3=0.20, dropout_dec2=0.15):
        super().__init__()
        f = init_features

        self.enc1 = ResidualBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(f * 8, f * 16)
        self.bottleneck_dropout = nn.Dropout2d(p=dropout_bottleneck)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.att4 = AttentionGate(f * 8, f * 8, f * 4)
        self.dec4 = ResidualBlock(f * 16, f * 8)
        self.dec4_dropout = nn.Dropout2d(p=dropout_dec4)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.att3 = AttentionGate(f * 4, f * 4, f * 2)
        self.dec3 = ResidualBlock(f * 8, f * 4)
        self.dec3_dropout = nn.Dropout2d(p=dropout_dec3)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.att2 = AttentionGate(f * 2, f * 2, f)
        self.dec2 = ResidualBlock(f * 4, f * 2)
        self.dec2_dropout = nn.Dropout2d(p=dropout_dec2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.att1 = AttentionGate(f, f, f // 2)
        self.dec1 = ResidualBlock(f * 2, f)

        self.head = nn.Conv2d(f, num_classes, 1)
        self.ds4 = nn.Conv2d(f * 8, num_classes, 1)
        self.ds3 = nn.Conv2d(f * 4, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        bn = self.bottleneck_dropout(self.bottleneck(self.pool4(e4)))

        d4 = self.up4(bn)
        d4 = self.dec4(torch.cat([d4, self.att4(g=d4, x=e4)], dim=1))
        d4 = self.dec4_dropout(d4)

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(g=d3, x=e3)], dim=1))
        d3 = self.dec3_dropout(d3)

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(g=d2, x=e2)], dim=1))
        d2 = self.dec2_dropout(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(g=d1, x=e1)], dim=1))

        out = self.head(d1)

        if self.training:
            ds4_out = F.interpolate(self.ds4(d4), size=out.shape[2:], mode="bilinear", align_corners=False)
            ds3_out = F.interpolate(self.ds3(d3), size=out.shape[2:], mode="bilinear", align_corners=False)
            return out, ds4_out, ds3_out
        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits.view(-1), targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]))
        self.dice_fn = DiceLoss()
        self.focal_fn = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    def _head_loss(self, logits, targets):
        return (BCE_WEIGHT * self.bce_fn(logits, targets)
                + DICE_WEIGHT * self.dice_fn(logits, targets)
                + FOCAL_WEIGHT * self.focal_fn(logits, targets))

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            w_main, w_ds4, w_ds3 = DS_WEIGHTS
            main_out, ds4_out, ds3_out = outputs
            return (w_main * self._head_loss(main_out, targets)
                    + w_ds4 * self._head_loss(ds4_out, targets)
                    + w_ds3 * self._head_loss(ds3_out, targets))
        return self._head_loss(outputs, targets)


class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.tn = self.fn = 0

    @torch.no_grad()
    def update(self, logits, targets):
        preds = (torch.sigmoid(logits) >= self.threshold).float()
        targets = targets.float()
        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.tn += ((preds == 0) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()

    def compute(self):
        eps = 1e-7
        water_iou = self.tp / (self.tp + self.fp + self.fn + eps)
        bg_iou = self.tn / (self.tn + self.fp + self.fn + eps)
        return {
            "water_iou": water_iou,
            "mean_iou": (water_iou + bg_iou) / 2.0,
            "dice": 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps),
            "precision": self.tp / (self.tp + self.fp + eps),
            "recall": self.tp / (self.tp + self.fn + eps),
        }


def train_one_epoch(model, loader, criterion, optimizer, scaler, metrics):
    model.train()
    metrics.reset()
    total_loss, n = 0.0, 0
    for images, masks, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        metrics.update(outputs[0].detach(), masks)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1), metrics.compute()


@torch.no_grad()
def validate(model, loader, criterion, metrics):
    model.eval()
    metrics.reset()
    total_loss, n = 0.0, 0
    for images, masks, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, masks)
        metrics.update(outputs, masks)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1), metrics.compute()


def make_loaders(train_dataset, val_dataset, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    weights = torch.DoubleTensor([WATER_OVERSAMPLE if w else 1.0 for w in train_dataset.has_water])
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=g)
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin)
    return train_loader, val_loader


def run_experiment(seed, train_dataset, val_dataset, weights_dir):
    seed_everything(seed)
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, seed)

    model = ResidualAttentionUNet(
        in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, init_features=INIT_FEATURES,
        dropout_bottleneck=DROPOUT_BOTTLENECK, dropout_dec4=DROPOUT_DEC4,
        dropout_dec3=DROPOUT_DEC3, dropout_dec2=DROPOUT_DEC2,
    ).to(DEVICE)
    criterion = CombinedLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    swa_start = NUM_EPOCHS - SWA_EPOCHS
    warmup = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / WARMUP_EPOCHS))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, swa_start - WARMUP_EPOCHS), eta_min=ETA_MIN)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=SWA_ANNEAL_EPOCHS, anneal_strategy="cos")

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    train_metrics = SegmentationMetrics()
    val_metrics = SegmentationMetrics()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch(model, train_loader, criterion, optimizer, scaler, train_metrics)
        val_loss, val_m = validate(model, val_loader, criterion, val_metrics)

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        print(f"seed {seed} | epoch {epoch:03d}/{NUM_EPOCHS} | {time.time() - t0:.0f}s | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"train loss {train_loss:.4f} IoU {train_m['water_iou']:.4f} | "
              f"val loss {val_loss:.4f} IoU {val_m['water_iou']:.4f}")

    update_bn(train_loader, swa_model, device=DEVICE)
    val_loss, val_m = validate(swa_model, val_loader, criterion, val_metrics)
    print(f"seed {seed} | SWA model | val loss {val_loss:.4f} IoU {val_m['water_iou']:.4f} Dice {val_m['dice']:.4f}")

    torch.save({"model_state_dict": swa_model.module.state_dict(), "seed": seed, "val_metrics": val_m},
               os.path.join(weights_dir, f"swa_seed{seed}.pth"))

    result = {"seed": seed, "val_loss": val_loss}
    result.update({f"val_{k}": v for k, v in val_m.items()})
    return result


def main():
    weights_dir = os.path.join(OUTPUT_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    train_dataset = WaterDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, augment=True)
    val_dataset = WaterDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, augment=False)
    print(f"{len(train_dataset)} training patches, {len(val_dataset)} validation patches")

    results = [run_experiment(s, train_dataset, val_dataset, weights_dir) for s in SEEDS]

    summary = {}
    for key in results[0]:
        if key == "seed":
            continue
        vals = np.array([r[key] for r in results], dtype=float)
        summary[key] = {"mean": float(vals.mean()),
                        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0}
        print(f"{key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({"seeds": SEEDS, "per_seed": results, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
