#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""USM 锐化（Unsharp Mask）—— 用 OpenCV 模拟 Photoshop 的 USM 锐化。

思路：先做半径为 r 的高斯模糊得到「模糊版」，
    锐化 = 原图 + 数量 × (原图 − 模糊版)
细节层（原图与模糊版的差值）小于阈值的像素不参与锐化，保护平坦区域
不被放大噪点。三个参数均为 Photoshop 同款语义，常量在顶部调整：

    数量 USM_AMOUNT     50%（PS 刻度 1~500%，这里用小数表示）
    半径 USM_RADIUS     1 像素（对应高斯模糊的 σ）
    阈值 USM_THRESHOLD  0 色阶（PS 刻度 0~255；0 = 全图锐化，常用 2~20 抑制噪点）

作为模块被 main.py 集成，对 float32 [0,1] 图像工作。
"""
from __future__ import annotations

import cv2
import numpy as np

USM_AMOUNT = 0.5    # 数量 50%
USM_RADIUS = 1.0    # 半径 1 px
USM_THRESHOLD = 0   # 阈值 0 色阶


def usm_sharpen(img01: np.ndarray, amount: float = USM_AMOUNT,
                radius: float = USM_RADIUS,
                threshold: float = USM_THRESHOLD) -> np.ndarray:
    """(H,W,C) float32 [0,1] → USM 锐化后返回，形状不变，结果裁剪到 [0,1]。

    threshold 为 Photoshop 的色阶刻度（0~255），内部换算到 [0,1]；
    与蒙尘与划痕一致：任意通道差值超阈值即参与锐化。
    """
    gray_in = img01.shape[-1] == 1
    img = img01[..., 0] if gray_in else img01
    detail = img - cv2.GaussianBlur(img, (0, 0), radius)
    if threshold > 0:
        thr = threshold / 255.0
        mask = (np.any(np.abs(detail) > thr, axis=2) if detail.ndim == 3
                else np.abs(detail) > thr)
        out = img.copy()
        out[mask] = img[mask] + amount * detail[mask]
    else:
        out = img + amount * detail
    out = np.clip(out, 0.0, 1.0)
    return out[..., None] if gray_in else out
