#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TIFF_SUFFIXES = {".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description="批量重命名")
    p.add_argument("path", nargs="?", default=".",
                   help="处理目录（默认当前目录）")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="递归处理子目录")
    p.add_argument("--dry-run", action="store_true",
                   help="预览将做的改动")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.path)
    if not root.is_dir():
        sys.exit(f"目录不存在：{root}")

    pattern = "**/*" if args.recursive else "*"
    files = sorted(p for p in root.glob(pattern)
                   if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES)

    n_done = n_keep = n_collide = n_fail = 0
    for f in files:
        new_stem = f.stem.split("_", 1)[0]
        if "_" not in f.stem or not new_stem:
            n_keep += 1
            continue
        target = f.with_name(new_stem + f.suffix)
        if target.exists():
            print(f"[跳过] {f.name} → {target.name}：目标已存在")
            n_collide += 1
            continue
        if args.dry_run:
            print(f"[预览] {f.name} → {target.name}")
        else:
            try:
                f.rename(target)
            except OSError as e:  # 文件被占用等
                print(f"[失败] {f.name}：{e}")
                n_fail += 1
                continue
            print(f"[重命名] {f.name} → {target.name}")
        n_done += 1

    action = "预览" if args.dry_run else "重命名"
    print(f"\n共扫描 {len(files)} 个 TIFF：{action} {n_done}，"
          f"无需改名 {n_keep}，目标冲突跳过 {n_collide}，失败 {n_fail}。")


if __name__ == "__main__":
    main()
