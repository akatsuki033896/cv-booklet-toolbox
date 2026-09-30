#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT 去网纹（de-screen）核心模块 —— 面向 600 dpi 固定尺寸扫描件
================================================================

把人工流程自动化：

    原始扫描 → FFT → 定位频谱周期性亮点 → 高斯 notch 滤波 → 逆 FFT → 输出

亮点定位的两种方式
------------------
1) 写死（量产推荐）：扫描件固定 12cm × 24cm @ 600 dpi ≈ 2835 × 5669 px，
   网纹频率只由印刷参数决定，与图像内容无关：
       f = LPI / 600  (cycles/px)，方向 = 网角 θ
       亮点位于 ±f·(cosθ, sinθ)，以及 k·f 谐波处
   用 fixed_centers(LPI, 网角) 生成中心点，再传给 descreen(centers=...)。
2) 自动检测（默认）：log 幅度谱减去低频背景后找局部极大值。
   拿到新印刷品时可先用自动模式跑一遍，控制台会把每个亮点换算成
   「LPI + 网角」打印出来，确定参数后即可切换到写死模式。

命令行入口见 main.py；本模块只负责滤波本身。
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy import ndimage

from utils import rfft2, irfft2, luma, reflect_pad

# =====================================================================
# 硬编码参数区 —— 扫描件固定 600 dpi、约 12cm × 24cm（≈2835 × 5669 px）
# =====================================================================
SCAN_DPI = 600            # 扫描分辨率
SCREEN_LPI = 150.0        # 印刷网线数（写死位置时改这里；常见 65~175）
SCREEN_ANGLE_DEG = 45.0   # 网屏角度，从水平方向起算（写死位置时改这里）
N_HARMONICS = 2           # 除基频 f 外，还压制的谐波个数（2f、3f…）
NOTCH_SIGMA = 0.006       # notch 半径，单位 cycles/px（谱内约 17~35 px）

# 自动检测参数
AUTO_PEAKS = 12           # 最多检测的亮点数
AUTO_FMIN = 0.05          # 忽略离 DC 更近的频率（对应约 30 LPI 以下）
AUTO_FMAX = 0.48          # 忽略接近 Nyquist 的频率
AUTO_MIN_SEP = 0.012      # 亮点之间的最小间距 (cycles/px)
AXIS_GUARD = 0.006        # 检测时屏蔽沿频率轴的边缘泄漏亮带 (cycles/px)
AUTO_PROM_RATIO = 0.15    # 突出度阈值：残差需 >= 最强亮点的 15%（滤除旁瓣/噪点）

# ---------------------------------------------------------------------
# notch 滤波
# ---------------------------------------------------------------------
def wrapped_stored(vr: float, vc: float) -> list[tuple[float, float]]:
    """(vr,vc) 及其共轭镜像 (-vr,-vc) 中，落在 rfft 存储半平面（列频率 >= 0）
    的频率坐标。内部统一约定：(vr=行/垂直频率, vc=列/水平频率)，单位 cycles/px。"""
    out = []
    for gr, gc in ((vr, vc), (-vr, -vc)):
        wc = gc % 1.0
        if wc <= 0.5 + 1e-6:
            out.append(((gr + 0.5) % 1.0 - 0.5, wc))
    return out


def notch_mask(shape_rfft: tuple[int, int], width_full: int,
               centers, sigma: float) -> np.ndarray:
    """乘性高斯 notch 模板：亮点处 → 0，远处 → 1，DC 与低频不受影响。"""
    R, W = shape_rfft
    fr = np.fft.fftfreq(R)[:, None].astype(np.float32)
    fc = np.fft.rfftfreq(width_full)[None, :].astype(np.float32)
    mask = np.ones((R, W), dtype=np.float32)
    two_s2 = 2.0 * sigma * sigma
    for vr, vc in centers:
        for wr, wc in wrapped_stored(vr, vc):
            d2 = (fr - wr) ** 2 + (fc - wc) ** 2
            mask *= 1.0 - np.exp(-d2 / two_s2)
    return mask


def fixed_centers(lpi: float, angle_deg: float, harmonics: int, dpi: int):
    """写死模式：由 LPI/网角换算亮点频率（含 k·f 谐波，自动含共轭镜像）。"""
    f = lpi / dpi
    th = math.radians(angle_deg)
    centers = []
    for k in range(1, harmonics + 2):
        vr = k * f * math.sin(th)
        vc = k * f * math.cos(th)
        if abs(vr) >= 0.499 or abs(vc) >= 0.499:
            continue  # 超出 Nyquist 的谐波没有意义
        centers.append((vr, vc))
    return centers


# ---------------------------------------------------------------------
# 自动检测频谱亮点
# ---------------------------------------------------------------------
def auto_detect_centers(mag: np.ndarray, width_full: int, n_peaks: int,
                        fmin: float, fmax: float, min_sep: float,
                        axis_guard: float = AXIS_GUARD):
    """在对数幅度谱上：减去大尺度背景 → 频带内找局部极大 → 贪心选 top-N。
    只在 rfft 存储的半平面里找，每个共轭对至少有一个成员落在其中。"""
    resid = np.log1p(mag / (mag.max() + 1e-12))
    resid -= cv2.GaussianBlur(resid, (0, 0), 20)  # 去掉 1/f 背景，突出亮点

    R, W = mag.shape
    fr = np.fft.fftfreq(R)
    fc = np.fft.rfftfreq(width_full)
    rad2 = fr[:, None] ** 2 + fc[None, :] ** 2
    ok = (rad2 >= fmin ** 2) & (rad2 <= fmax ** 2)
    # 屏蔽沿两条频率轴的亮带（图像边缘/均值泄漏造成的假亮点）
    ok &= (np.abs(fr[:, None]) >= axis_guard) & (np.abs(fc[None, :]) >= axis_guard)
    resid = np.where(ok, resid, -1e9)

    ksize = 2 * max(3, int(round(min_sep * R))) + 1
    localmax = resid >= ndimage.maximum_filter(resid, size=ksize) - 1e-6
    ys, xs = np.nonzero(localmax & ok)
    vals = resid[ys, xs]

    if len(vals) == 0:
        return []
    order = np.argsort(vals)[::-1]
    # 突出度阈值：取「鲁棒噪声底」与「最强亮点的固定比例」中的较大者，
    # 滤掉真亮点附近的旁瓣脊和随机噪点（实测两者相差 3 个数量级）。
    band_vals = resid[ok]
    med = float(np.median(band_vals))
    mad = 1.4826 * float(np.median(np.abs(band_vals - med)))
    thresh = max(med + 8.0 * mad, AUTO_PROM_RATIO * float(vals[order[0]]))

    selected: list[tuple[float, float]] = []
    for i in order:
        if vals[i] < thresh:
            break
        cand = (float(fr[int(ys[i])]), float(fc[int(xs[i])]))
        if all((cand[0] - s[0]) ** 2 + (cand[1] - s[1]) ** 2 >= min_sep ** 2
               for s in selected):
            selected.append(cand)
        if len(selected) >= n_peaks:
            break
    return selected


def detect_centers(img01: np.ndarray):
    gray_p = reflect_pad(luma(img01))
    F = rfft2(gray_p)
    mag = np.abs(F).astype(np.float32)
    return auto_detect_centers(mag, gray_p.shape[1], AUTO_PEAKS,
                               AUTO_FMIN, AUTO_FMAX, AUTO_MIN_SEP)


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def descreen(img01: np.ndarray, centers: list[tuple[float, float]] | None = None,
             sigma: float = NOTCH_SIGMA, save_prefix: Path | None = None):
    """FFT → notch → 逆 FFT。centers 为空时自动检测亮点，sigma 默认 NOTCH_SIGMA。
    返回 (结果[0,1], 实际使用的centers, 每组抑制dB)。"""
    h, w = img01.shape[:2]
    gray_p = reflect_pad(luma(img01))
    Rr, Wr = rfft2(gray_p).shape
    width_full = gray_p.shape[1]

    if not centers:
        centers = detect_centers(img01)

    mask = notch_mask((Rr, Wr), width_full, centers, sigma)

    out = np.empty_like(img01)
    for c in range(img01.shape[2]):
        ch_p = reflect_pad(img01[..., c])
        F = rfft2(ch_p)
        F *= mask
        out[..., c] = irfft2(F, ch_p.shape)[:h, :w]

    # 每个 notch 位置的能量抑制量（正 dB 越大越好）
    mag_before = np.abs(rfft2(gray_p)).astype(np.float32)
    mag_after = np.abs(rfft2(reflect_pad(luma(out)))).astype(np.float32)
    fr = np.fft.fftfreq(Rr)[:, None].astype(np.float32)
    fc = np.fft.rfftfreq(width_full)[None, :].astype(np.float32)
    gains = []
    for vr, vc in centers:
        sel = np.zeros((Rr, Wr), dtype=bool)
        for wr, wc in wrapped_stored(vr, vc):
            sel |= ((fr - wr) ** 2 + (fc - wc) ** 2) <= (2 * sigma) ** 2
        e0 = float((mag_before[sel] ** 2).sum())
        e1 = float((mag_after[sel] ** 2).sum())
        gains.append(-10 * math.log10((e1 + 1e-30) / (e0 + 1e-30)))

    if save_prefix is not None:
        save_spectrum_png(Path(str(save_prefix) + "_spectrum_before.png"),
                          mag_before, centers, width_full)
        save_spectrum_png(Path(str(save_prefix) + "_spectrum_after.png"),
                          mag_after, centers, width_full)
    return np.clip(out, 0.0, 1.0), centers, gains


def save_spectrum_png(path: Path, mag: np.ndarray, centers, width_full: int,
                      target_h: int = 1400) -> None:
    """保存居中的对数幅度谱（DC 在中心，横轴=水平频率，纵轴=垂直频率），
    并用白圈标出 notch 位置，便于人工核对。"""
    logmag = np.log1p(mag / (mag.max() + 1e-12))
    cent = np.fft.fftshift(logmag, axes=0)
    # 实图像频谱关于 DC 点对称：负频率半平面 = 行列同时翻转（点对称镜像）
    full = np.concatenate([cent[::-1, :0:-1], cent], axis=1)
    R, WT = full.shape
    lo, hi = np.percentile(full, (0.2, 99.8))
    vis = np.clip((full - lo) / max(hi - lo, 1e-9), 0, 1)
    vis = (vis * 255).astype(np.uint8)
    sw = max(1, int(round(WT * target_h / R)))
    vis = cv2.resize(vis, (sw, target_h), interpolation=cv2.INTER_AREA)
    sy, sx = target_h / R, sw / WT
    for vr, vc in centers:
        for gr, gc in ((vr, vc), (-vr, -vc)):  # 共轭镜像也画圈
            row = int(round((gr + 0.5) * R * sy))
            col = int(round((gc + 0.5) * (WT - 1) * sx))
            if 0 <= row < target_h and 0 <= col < sw:
                cv2.circle(vis, (col, row), 10, 255, 2)
    cv2.imwrite(str(path), vis)


def describe_center(cen: tuple[float, float]) -> str:
    vr, vc = cen
    f = math.hypot(vr, vc)
    ang = math.degrees(math.atan2(vr, vc))
    return f"(f={f:.4f} c/px ≈ {f * SCAN_DPI:6.1f} LPI @ {ang:6.1f}°)"


# ---------------------------------------------------------------------
# 自测：合成一张带网纹的模拟扫描件
# ---------------------------------------------------------------------
def make_synthetic_input(path: Path) -> Path:
    if path.exists():
        return path
    W, H = 2835, 5669  # 12cm × 24cm @ 600 dpi
    rng = np.random.default_rng(7)
    small = rng.standard_normal((H // 16, W // 16)).astype(np.float32)
    photo = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    photo = cv2.GaussianBlur(photo, (0, 0), 6)
    photo = np.clip(photo / (3.0 * photo.std() + 1e-9) * 0.35 + 0.5, 0.02, 0.98)

    f = SCREEN_LPI / SCAN_DPI
    th = math.radians(SCREEN_ANGLE_DEG)
    x = np.arange(W, dtype=np.float32)
    u = 2 * np.pi * f * math.cos(th) * x
    v = 2 * np.pi * f * math.sin(th) * np.arange(H, dtype=np.float32)
    screen = np.cos(v[:, None] + u[None, :])  # SCREEN_ANGLE_DEG° 网纹

    scanned = np.clip(photo - 0.20 * screen, 0.0, 1.0)
    arr = np.round(scanned * 255).astype(np.uint8)
    try:
        tifffile.imwrite(str(path), arr, compression="lzw")
    except KeyError:  # 缺 imagecodecs 时退回 OpenCV
        cv2.imwrite(str(path), arr)
    print(f"[自测] 已生成模拟扫描件 {path}"
          f"（{SCREEN_LPI:.0f} LPI @ {SCREEN_ANGLE_DEG:.0f}°，"
          f"f={f:.4f} c/px，期望亮点 (±{f * math.cos(th):.4f}, ±{f * math.sin(th):.4f})）")
    return path


def save_compare_png(path: Path, before01: np.ndarray, after01: np.ndarray,
                     size: int = 560) -> None:
    """并排保存中间裁剪块，方便肉眼对比。"""
    h, w = before01.shape[:2]
    y, x = h // 2, w // 2
    s = size // 2
    a = before01[y - s:y + s, x - s:x + s]
    b = after01[y - s:y + s, x - s:x + s]
    pair = np.concatenate([a, b], axis=1)
    if pair.shape[-1] == 1:
        pair = pair[..., 0]
    else:
        pair = pair[..., ::-1]  # RGB → BGR
    cv2.imwrite(str(path), np.round(np.clip(pair, 0, 1) * 255).astype(np.uint8))
