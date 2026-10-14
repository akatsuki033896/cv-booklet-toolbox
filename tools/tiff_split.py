#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tifffile


def parse_args():
    p = argparse.ArgumentParser(description="多页 TIFF 切分")
    p.add_argument("input", help="源 TIFF 文件")
    p.add_argument("-o", "--output", help="输出目录（默认与源文件同目录）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    if not inp.is_file():
        sys.exit(f"源文件不存在：{inp}")
    out_dir = Path(args.output) if args.output else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(inp) as tif:
        n = len(tif.pages)
        print(f"[输入] {inp.name}  共 {n} 页")
        width = max(2, len(str(n)))
        n_done = n_skip = 0
        for i, page in enumerate(tif.pages, 1):
            out_path = out_dir / f"bk{i:0{width}d}{inp.suffix}"
            if out_path.exists():
                print(f"[跳过] {out_path.name}：目标已存在")
                n_skip += 1
                continue
            arr = page.asarray()
            try:
                tifffile.imwrite(out_path, arr, compression="lzw")
            except KeyError:
                tifffile.imwrite(out_path, arr)
            print(f"[输出] {out_path}")
            n_done += 1

    print(f"\n共 {n} 页：输出 {n_done}，目标冲突跳过 {n_skip} → {out_dir}")


if __name__ == "__main__":
    main()
