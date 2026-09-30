#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT 去网纹（de-screen）CLI —— 面向 600 dpi 固定尺寸扫描件
=========================================================

    原始扫描 → FFT → 自动定位频谱周期性亮点 → 高斯 notch 滤波 → 逆 FFT → 输出

用法
----
    python main.py scan.tif                         # 去网纹
    python main.py scan.tif --blur                  # 去网纹 + 高斯模糊
    python main.py scan.tif --noise                 # 去网纹 + 高斯模糊 + 蒙尘与划痕
    python main.py scan.tif -o out.tif --spectrum   # 指定输出路径、保存频谱 PNG

输出（绝不改动原文件，全部是新文件）：
    <输入名>_descreen.tif          处理结果（与原图同尺寸、同位深）
    <输入名>_spectrum_before.png   滤波前频谱，白圈 = notch 位置（仅 --spectrum 时生成）
    <输入名>_spectrum_after.png    滤波后频谱（仅 --spectrum 时生成）

各步骤实现在独立模块中，参数（常量）也在各模块顶部调整：
    notch_filter.py   notch 去网纹（网线数/网角/notch 半径/自动检测阈值）
    gaussian_blur.py  高斯模糊后处理（σ）
    dust_scratch.py   蒙尘与划痕（半径/阈值）
    utils.py          图像读写、FFT 等基础工具
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dust_scratch import DUST_RADIUS, DUST_THRESHOLD, dust_and_scratches
from gaussian_blur import GAUSS_SIGMA, gaussian_blur
from notch_filter import SCAN_DPI, describe_center, descreen
from utils import load_image, save_image

def parse_args():
    p = argparse.ArgumentParser(description="FFT notch 去网纹")
    p.add_argument("input", help="输入 TIFF（600 dpi 扫描件）")
    p.add_argument("-o", "--output", help="输出路径（默认 <输入名>_descreen.tif）")
    p.add_argument("--spectrum", action="store_true",
                   help="保存滤波前后的频谱 PNG（默认不生成）")
    p.add_argument("--blur", action="store_true",
                   help=f"去网纹后叠加高斯模糊（σ={GAUSS_SIGMA}）")
    p.add_argument("--noise", action="store_true",
                   help=f"去网纹 + 高斯模糊 + 蒙尘与划痕"
                        f"（r={DUST_RADIUS}, t={DUST_THRESHOLD}）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"输入文件不存在：{inp}")

    img01, dtype = load_image(inp)
    h, w = img01.shape[:2]
    print(f"[输入] {inp.name}  {w}×{h} px  dtype={np.dtype(dtype).name}")

    out_path = Path(args.output) if args.output else inp.with_name(
        inp.stem + "_descreen.tif")
    if out_path.resolve() == inp.resolve():
        sys.exit("输出路径与输入相同：拒绝覆盖原文件。")

    steps = ["notch 去网纹"]
    if args.blur or args.noise:
        steps.append(f"高斯模糊(σ={GAUSS_SIGMA})")
    if args.noise:
        steps.append(f"蒙尘与划痕(r={DUST_RADIUS}, t={DUST_THRESHOLD})")
    print("[流程] " + " → ".join(steps)
          + "（亮点自动检测，确定参数后可在 notch_filter.py 写死）")

    prefix = inp.with_name(inp.stem) if args.spectrum else None
    result, centers, gains = descreen(img01, save_prefix=prefix)
    if args.blur or args.noise:
        result = gaussian_blur(result)
    if args.noise:
        result = dust_and_scratches(result)
    save_image(out_path, result, dtype, SCAN_DPI)

    if centers:
        print(f"[检测] 共 {len(centers)} 组：")
        for cen, g in zip(centers, gains):
            print(f"    (vr={cen[0]:+.4f}, vc={cen[1]:+.4f}) "
                  f"{describe_center(cen)}  抑制 {g:5.1f} dB")
    else:
        print("[警告] 未找到任何周期性亮点，输出与输入基本相同。")

    print(f"[输出] {out_path}（同尺寸/同位深；原文件未做任何改动）")
    if prefix is not None:
        print(f"       {prefix}_spectrum_before.png / _spectrum_after.png")


if __name__ == "__main__":
    main()
