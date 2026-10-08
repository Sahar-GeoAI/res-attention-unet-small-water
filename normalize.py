import os
import sys

import numpy as np
import rasterio

OPTICAL_P99 = [2031.09, 1729.96, 1325.78, 4952.32, 3379.47]
AWEISH_P1, AWEISH_P99 = -9067.03, 1370.92
MBWI_P1, MBWI_P99 = -8097.45, 401.69
MNDWI_P1, MNDWI_P99 = -0.718, 0.863
NDVI_P1, NDVI_P99 = -0.692, 0.922
TWI_P1, TWI_P99 = 2.763, 15.472
ASINH_SCALE = 1.0
SLOPE_MAX = 30.0
N_BANDS = 11
FILL_VALUE = 0.0


def minmax(band, vmin, vmax):
    if vmax == vmin:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


def asinh_minmax(band, p1, p99, s=ASINH_SCALE):
    return minmax(np.arcsinh(band / s), float(np.arcsinh(p1 / s)), float(np.arcsinh(p99 / s)))


def normalize_stack(data):
    out = np.empty_like(data, dtype=np.float32)
    for b in range(5):
        out[b] = minmax(data[b], 0.0, OPTICAL_P99[b])
    out[5] = asinh_minmax(data[5], AWEISH_P1, AWEISH_P99)
    out[6] = asinh_minmax(data[6], MBWI_P1, MBWI_P99)
    out[7] = minmax(data[7], MNDWI_P1, MNDWI_P99)
    out[8] = minmax(data[8], NDVI_P1, NDVI_P99)
    out[9] = np.clip(data[9], 0.0, SLOPE_MAX) / SLOPE_MAX
    out[10] = minmax(data[10], TWI_P1, TWI_P99)
    return out


def normalize_file(in_path, out_path):
    with rasterio.open(in_path) as src:
        if src.count != N_BANDS:
            print(f"Skipped {os.path.basename(in_path)}: {src.count} bands")
            return False
        data = src.read().astype(np.float32)
        nodata = src.nodata
        profile = src.profile.copy()

    invalid = ~np.isfinite(data).all(axis=0)
    if nodata is not None:
        invalid |= np.any(data == nodata, axis=0)

    out = normalize_stack(data)
    out[:, invalid] = FILL_VALUE

    profile.update(dtype="float32", count=N_BANDS, compress="deflate", predictor=3, nodata=None)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out)
            dst.write_mask((~invalid).astype(np.uint8) * 255)
    return True


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff")))
    done = sum(normalize_file(os.path.join(input_dir, f), os.path.join(output_dir, f)) for f in files)
    print(f"{done}/{len(files)} files written to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python normalize.py INPUT_DIR OUTPUT_DIR")
    main(sys.argv[1], sys.argv[2])
