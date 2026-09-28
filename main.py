#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT 去网纹（de-screen）最小 Demo —— 面向 600 dpi 固定尺寸扫描件
================================================================

把人工流程自动化：

    原始扫描 → FFT → 定位频谱周期性亮点 → 高斯 notch 滤波 → 逆 FFT → 输出

亮点定位的两种方式
------------------
1) 写死（量产推荐）：扫描件固定 12cm × 24cm @ 600 dpi ≈ 2835 × 5669 px，
   网纹频率只由印刷参数决定，与图像内容无关：
       f = LPI / 600  (cycles/px)，方向 = 网角 θ
       亮点位于 ±f·(cosθ, sinθ)，以及 k·f 谐波处
   修改文件顶部常量 SCREEN_LPI / SCREEN_ANGLE_DEG，或用 --lpi/--angle。
2) 自动检测（默认）：log 幅度谱减去低频背景后找局部极大值。
   拿到新印刷品时可先用自动模式跑一遍，控制台会把每个亮点换算成
   「LPI + 网角」打印出来，直接填到常量里即可写死。

用法
----
    python main.py scan.tif                          # 自动检测亮点
    python main.py scan.tif --lpi 150 --angle 45     # 写死网线数/网角
    python main.py scan.tif --notch 0.177,-0.177     # 手动指定频率坐标，可重复
    python main.py --selftest                        # 合成带网纹的图并验证

输出（绝不改动原文件，全部是新文件）：
    <输入名>_descreen.tif          去网纹结果（与原图同尺寸、同位深）
    <输入名>_spectrum_before.png   滤波前频谱（白圈 = notch 位置）
    <输入名>_spectrum_after.png    滤波后频谱
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("缺少 opencv-python，请先安装：python -m pip install opencv-python")

try:
    from scipy import fft as spfft
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

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
# 基础工具
# ---------------------------------------------------------------------
def next_fast_len(n: int) -> int:
    """对 FFT 友好的长度（5/7-smooth）。高度 5669 是素数，直接变换会很慢。"""
    if HAS_SCIPY:
        return spfft.next_fast_len(n)
    m = n
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def rfft2(a: np.ndarray) -> np.ndarray:
    return spfft.rfft2(a, workers=-1) if HAS_SCIPY else np.fft.rfft2(a)


def irfft2(F: np.ndarray, s: tuple[int, int]) -> np.ndarray:
    return spfft.irfft2(F, s=s, workers=-1) if HAS_SCIPY else np.fft.irfft2(F, s=s)


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
    if HAS_TIFFFILE:
        arr = tifffile.imread(str(path))
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        sys.exit(f"无法读取图像：{path}")
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


def save_image(path: Path, img01: np.ndarray, dtype) -> None:
    if dtype == np.uint8:
        out = np.round(np.clip(img01, 0, 1) * 255).astype(np.uint8)
    elif dtype == np.uint16:
        out = np.round(np.clip(img01, 0, 1) * 65535).astype(np.uint16)
    else:
        out = img01.astype(np.float32)
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    if HAS_TIFFFILE:
        try:
            tifffile.imwrite(str(path), out, compression="lzw",
                             resolution=(SCAN_DPI, SCAN_DPI))
        except KeyError:  # 缺 imagecodecs 时退回无压缩
            tifffile.imwrite(str(path), out, resolution=(SCAN_DPI, SCAN_DPI))
    else:
        cv2.imwrite(str(path), out)


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
    if HAS_SCIPY:
        localmax = resid >= ndimage.maximum_filter(resid, size=ksize) - 1e-6
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        localmax = resid >= cv2.dilate(resid, kernel) - 1e-6
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
def descreen(img01: np.ndarray, centers, sigma: float, save_prefix: Path | None):
    """FFT → notch → 逆 FFT。返回 (结果[0,1], 实际使用的centers, 每组抑制dB)。"""
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
    except (KeyError, NameError):
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


# ---------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="FFT notch 去网纹 demo（只写新文件，不修改原文件）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", nargs="?", help="输入 TIFF（600 dpi 扫描件）")
    p.add_argument("-o", "--output", help="输出路径（默认 <输入名>_descreen.tif）")
    p.add_argument("--lpi", type=float, help="印刷网线数（写死模式，配合 --angle）")
    p.add_argument("--angle", type=float, help="网屏角度，度（写死模式）")
    p.add_argument("--notch", action="append", default=[], metavar="FX,FY",
                   help="手动追加 notch：水平,垂直频率 (cycles/px)，可多次")
    p.add_argument("--peaks", type=int, default=AUTO_PEAKS,
                   help="自动模式最多检测的亮点数")
    p.add_argument("--sigma", type=float, default=NOTCH_SIGMA,
                   help="notch 半径 (cycles/px)")
    p.add_argument("--no-spectrum", action="store_true", help="不保存频谱 PNG")
    p.add_argument("--selftest", action="store_true",
                   help="生成合成网纹扫描件并运行验证")
    return p.parse_args()


def main() -> None:
    global AUTO_PEAKS
    args = parse_args()
    AUTO_PEAKS = args.peaks

    if args.selftest:
        inp = make_synthetic_input(Path("selftest_scan.tif"))
    elif args.input:
        inp = Path(args.input)
    else:
        sys.exit("请给出输入 TIFF 路径，或使用 --selftest 自测。")
    if not inp.exists():
        sys.exit(f"输入文件不存在：{inp}")

    img01, dtype = load_image(inp)
    h, w = img01.shape[:2]
    print(f"[输入] {inp.name}  {w}×{h} px  dtype={np.dtype(dtype).name}")

    # 收集写死的 notch；为空则自动检测
    centers = []
    if args.lpi is not None or args.angle is not None:
        lpi = args.lpi if args.lpi is not None else SCREEN_LPI
        ang = args.angle if args.angle is not None else SCREEN_ANGLE_DEG
        centers += fixed_centers(lpi, ang, N_HARMONICS, SCAN_DPI)
        print(f"[写死] LPI={lpi:g}，网角={ang:g}° → {len(centers)} 组（f 及谐波）")
    for s in args.notch:
        fx, fy = map(float, s.split(","))
        centers.append((fy, fx))  # 内部约定 (行/垂直, 列/水平)
        print(f"[写死] 追加 notch：水平 {fx:+.4f} / 垂直 {fy:+.4f} c/px")
    if not centers:
        print("[模式] 自动检测频谱亮点（确定参数后可用 --lpi/--angle 写死）")

    out_path = Path(args.output) if args.output else inp.with_name(
        inp.stem + "_descreen.tif")
    if out_path.resolve() == inp.resolve():
        sys.exit("输出路径与输入相同：拒绝覆盖原文件。")

    prefix = None if args.no_spectrum else inp.with_name(inp.stem)
    result, centers, gains = descreen(img01, centers, args.sigma, prefix)
    save_image(out_path, result, dtype)

    if centers:
        print(f"[{'检测' if not (args.lpi or args.angle is not None or args.notch) else 'notch'}] "
              f"共 {len(centers)} 组：")
        for cen, g in zip(centers, gains):
            print(f"    (vr={cen[0]:+.4f}, vc={cen[1]:+.4f}) "
                  f"{describe_center(cen)}  抑制 {g:5.1f} dB")
    else:
        print("[警告] 未找到任何周期性亮点，输出与输入基本相同。")

    print(f"[输出] {out_path}（同尺寸/同位深；原文件未做任何改动）")
    if prefix is not None:
        print(f"       {prefix}_spectrum_before.png / _spectrum_after.png")
    if args.selftest:
        cmp_path = inp.with_name(inp.stem + "_compare.png")
        save_compare_png(cmp_path, img01, result)
        print(f"       {cmp_path}（左=带网纹原图，右=去网纹结果，中央裁剪）")


if __name__ == "__main__":
    main()
