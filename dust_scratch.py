#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒙尘与划痕（dust & scratches）—— 噪点/划痕清除。
流程：中值滤波得到邻域「标准」值 → 与原图求绝对差 → 差值超过阈值的像素（尘点/划痕）用中值替换，其余保留原图。
对 float32 [0,1] 图像工作。
"""
from __future__ import annotations

import cv2
import numpy as np

DUST_RADIUS = 1      # 半径：中心像素到邻域边缘的距离（PS 同款刻度），核大小 = 2r+1
DUST_THRESHOLD = 0   # 差值阈值（0~255，PS 同款刻度）；越大越保守，0 = 基本全按中值替换
# 0：全图中值滤波
# 清除：10-20

def dust_and_scratches(img01: np.ndarray, radius: int = DUST_RADIUS,
                       threshold: float = DUST_THRESHOLD) -> np.ndarray:
    """(H,W,C) float32 [0,1] → 去尘/划痕后返回，形状不变。

    与彩色图一致：任意通道差值超阈值即替换。
    注意：cv2.medianBlur 的 float32 输入只支持 3/5 核（即 radius ≤ 2）。
    """
    kernel_size = radius * 2 + 1
    gray_in = img01.shape[-1] == 1
    img = img01[..., 0] if gray_in else img01  # medianBlur 对 (H,W,1) 会返回 (H,W)
    median = cv2.medianBlur(img, kernel_size)
    diff = cv2.absdiff(img, median)
    thr = threshold / 255.0
    mask = np.any(diff > thr, axis=2) if diff.ndim == 3 else diff > thr
    out = img.copy()
    out[mask] = median[mask]
    return out[..., None] if gray_in else out
