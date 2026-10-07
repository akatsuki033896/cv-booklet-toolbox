#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from USM_sharpen import USM_AMOUNT, USM_RADIUS, USM_THRESHOLD, usm_sharpen
from dust_scratch import DUST_RADIUS, DUST_THRESHOLD, dust_and_scratches
from gaussian_blur import GAUSS_SIGMA, gaussian_blur
from notch_filter import SCAN_DPI, describe_center, descreen
from utils import load_image, save_image

# 步骤 → (默认输出后缀, 描述)
STEPS = {
    "descreen": ("_descreen", "notch 去网纹"),
    "blur": ("_blur", f"高斯模糊(σ={GAUSS_SIGMA})"),
    "noise": ("_noise", f"蒙尘与划痕(r={DUST_RADIUS}, t={DUST_THRESHOLD})"),
    "sharpen": ("_sharpen", f"USM 锐化(数量{USM_AMOUNT * 100:.0f}%, "
                            f"半径{USM_RADIUS:g}, 阈值{USM_THRESHOLD})"),
}


def parse_args():
    p = argparse.ArgumentParser(description="FFT notch 去网纹")
    p.add_argument("input", help="输入 TIFF")
    p.add_argument("-o", "--output",
                   help="输出路径（默认 <输入名><步骤后缀>.tif）")
    p.add_argument("--spectrum", action="store_true",
                   help="保存滤波前后的频谱")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--blur", action="store_true",
                   help=f"执行高斯模糊（σ={GAUSS_SIGMA}）")
    g.add_argument("--noise", action="store_true",
                   help=f"执行蒙尘与划痕（r={DUST_RADIUS}, t={DUST_THRESHOLD}）")
    g.add_argument("--sharpen", action="store_true",
                   help=f"执行 USM 锐化（数量{USM_AMOUNT * 100:.0f}%, "
                        f"半径{USM_RADIUS:g}, 阈值{USM_THRESHOLD}）")
    args = p.parse_args()
    if (args.blur or args.noise or args.sharpen) and args.spectrum:
        p.error("--spectrum 只配合去网纹步骤使用")
    return args


def main() -> None:
    args = parse_args()
    step = ("sharpen" if args.sharpen else
            "noise" if args.noise else
            "blur" if args.blur else "descreen")
    suffix, desc = STEPS[step]

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"输入文件不存在：{inp}")

    img01, dtype = load_image(inp)
    h, w = img01.shape[:2]
    print(f"[输入] {inp.name}  {w}×{h} px  dtype={np.dtype(dtype).name}")

    out_path = Path(args.output) if args.output else inp.with_name(
        inp.stem + suffix + ".tif")
    if out_path.resolve() == inp.resolve():
        sys.exit("输出路径与输入相同：拒绝覆盖原文件。")

    print(f"[步骤] {desc}")

    if step == "descreen":
        print("（亮点自动检测，确定参数后可在 notch_filter.py 写死）")
        prefix = inp.with_name(inp.stem) if args.spectrum else None
        result, centers, gains = descreen(img01, save_prefix=prefix)
        if centers:
            print(f"[检测] 共 {len(centers)} 组：")
            for cen, g in zip(centers, gains):
                print(f"    (vr={cen[0]:+.4f}, vc={cen[1]:+.4f}) "
                      f"{describe_center(cen)}  抑制 {g:5.1f} dB")
        else:
            print("[警告] 未找到任何周期性亮点，输出与输入基本相同。")
        if prefix is not None:
            print(f"       {prefix}_spectrum_before.png / _spectrum_after.png")
    elif step == "blur":
        result = gaussian_blur(img01)
    elif step == "sharpen":
        result = usm_sharpen(img01)
    else:
        result = dust_and_scratches(img01)

    save_image(out_path, result, dtype, SCAN_DPI)
    print(f"[输出] {out_path}（同尺寸/同位深；原文件未做任何改动）")


if __name__ == "__main__":
    main()
