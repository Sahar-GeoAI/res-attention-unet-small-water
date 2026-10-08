# Residual Attention U-Net for small water body mapping

Code and trained weights accompanying the paper:

> Sharifi, S., Andries, A., Morse, S., Murphy, R., Channon, Z., & Sperandio Nascimento, E. A multi-source deep learning approach to small water body mapping: disentangling the roles of learned super-resolution and LiDAR terrain. Under review.

The model segments surface water from an 11-channel input stack at 1 m pixel size, built from Sentinel-2 imagery super-resolved with S2DR3, four spectral water and vegetation indices, and slope and the Topographic Wetness Index derived from the Environment Agency LiDAR Composite DTM.

## Contents

| File | Description |
| --- | --- |
| `train.py` | Network definition and training procedure (four seeds, stochastic weight averaging) |
| `normalize.py` | Per-channel normalisation with the statistics of the training tiles |
| `weights/swa_seed{42,123,456,789}.pth` | Trained weights of the four models that form the final ensemble |

## Requirements

Python 3.10 or later with PyTorch 2.7 or later, NumPy, SciPy and rasterio.

```
pip install -r requirements.txt
```

## Input data

Each input is a GeoTIFF with 11 bands in the following order:

| Band | Channel | Band | Channel |
| --- | --- | --- | --- |
| 1 | Red | 7 | MBWI |
| 2 | Green | 8 | MNDWI |
| 3 | Blue | 9 | NDVI |
| 4 | NIR | 10 | Slope (degrees) |
| 5 | SWIR1 | 11 | TWI |
| 6 | AWEIsh | | |

Optical bands are Sentinel-2 Level-2A surface reflectance scaled by 10,000. AWEIsh and MBWI are computed from these scaled values. The sites, acquisition dates and LiDAR survey dates are listed in Supplementary Table S1 of the paper. Imagery is not included in this repository because it is openly available from the Copernicus Data Space Ecosystem and the Environment Agency.

## Usage

Normalise the input stacks:

```
python normalize.py INPUT_DIR OUTPUT_DIR
```

To train, place 256 × 256 patches of the normalised stacks and the matching binary masks in `data/train/images`, `data/train/masks`, `data/val/images` and `data/val/masks`, or edit the paths at the top of `train.py`, then run:

```
python train.py
```

To predict with the released weights:

```python
import torch
from train import ResidualAttentionUNet

models = []
for seed in [42, 123, 456, 789]:
    model = ResidualAttentionUNet(in_channels=11)
    ckpt = torch.load(f"weights/swa_seed{seed}.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    models.append(model.eval())

x = ...  # normalised input, tensor of shape (1, 11, 256, 256)
with torch.no_grad():
    prob = torch.stack([torch.sigmoid(m(x)) for m in models]).mean(0)
water = prob >= 0.5
```

The results reported in the paper were obtained on full tiles with overlapping windows and eight-fold test-time augmentation, as described in Section 2.4.

## Licence

The code is released under the MIT Licence.

## Contact

Sahar Sharifi, Centre for Environment and Sustainability, University of Surrey (s.sharifi@surrey.ac.uk)
"For questions, bug reports, or feedback, please open an issue in this repository."
