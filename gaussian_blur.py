#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import cv2
import numpy as np

GAUSS_SIGMA = 0.8  # 模糊强度（常用 0.5~1.2，过大会损失细节）


def gaussian_blur(img01: np.ndarray, sigma: float = GAUSS_SIGMA) -> np.ndarray:
    """(H,W,C) float32 [0,1] → 高斯模糊后返回，形状不变。"""
    gray_in = img01.shape[-1] == 1
    img = img01[..., 0] if gray_in else img01
    out = cv2.GaussianBlur(img, (0, 0), sigma)
    return out[..., None] if gray_in else out
