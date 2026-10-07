from pathlib import Path

import numpy as np
import tifffile
from scipy import fft as spfft


def next_fast_len(n: int) -> int:
    """对 FFT 友好的长度（5/7-smooth）。高度 5669 是素数，直接变换会很慢。"""
    return spfft.next_fast_len(n)


def rfft2(a: np.ndarray) -> np.ndarray:
    return spfft.rfft2(a, workers=-1)


def irfft2(F: np.ndarray, s: tuple[int, int]) -> np.ndarray:
    return spfft.irfft2(F, s=s, workers=-1)


def reflect_pad(a: np.ndarray) -> np.ndarray:
    """pad 到 FFT 友好尺寸；reflect 填充避免边缘跳变产生的频谱泄漏。"""
    R, C = next_fast_len(a.shape[0]), next_fast_len(a.shape[1])
    if (R, C) == a.shape:
        return a
    return np.pad(a, ((0, R - a.shape[0]), (0, C - a.shape[1])), mode="reflect")


def luma(img01: np.ndarray) -> np.ndarray:
    """(H,W,C) float32 [0,1] → 灰度 (H,W)，用于检测亮点。"""
    if img01.shape[-1] == 1:
        return img01[..., 0]
    return img01[..., 0] * 0.299 + img01[..., 1] * 0.587 + img01[..., 2] * 0.114


def load_image(path: Path):
    """读取 TIFF → float32 [0,1]。返回 (array(H,W,C), 原dtype)。"""
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        print("注意：疑似多页 TIFF，仅处理第 1 页")
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.dtype == np.uint8:
        scale, out_dtype = 255.0, np.uint8
    elif arr.dtype == np.uint16:
        scale, out_dtype = 65535.0, np.uint16
    else:
        scale, out_dtype = 1.0, np.float32
    return np.asarray(arr, dtype=np.float32) / scale, out_dtype


def save_image(path: Path, img01: np.ndarray, dtype, dpi: int) -> None:
    if dtype == np.uint8:
        out = np.round(np.clip(img01, 0, 1) * 255).astype(np.uint8)
    elif dtype == np.uint16:
        out = np.round(np.clip(img01, 0, 1) * 65535).astype(np.uint16)
    else:
        out = img01.astype(np.float32)
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    # 水平差分预测器只用于整数位深：16bit 连续调图像的 LZW 体积约减半，
    # 且属于 TIFF 6.0 标准Predictor=2，读取端通用；浮点数据不启用。
    predictor = dtype in (np.uint8, np.uint16) or None
    try:
        tifffile.imwrite(str(path), out, compression="lzw",
                         resolution=(dpi, dpi), predictor=predictor)
    except KeyError:  # 缺 imagecodecs 时退回无压缩
        tifffile.imwrite(str(path), out, resolution=(dpi, dpi))
