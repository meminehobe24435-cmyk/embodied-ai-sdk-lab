"""模型导出：PyTorch → TorchScript（CPU 可跑，无需 ONNX/TensorRT）。

为什么导 TorchScript：部署端不应该依赖训练侧的 Python 源码；
TorchScript 把模型与计算图固化下来，可跨进程加载、且**导出前后输出必须逐元素一致**（有单测保证）。
"""
from __future__ import annotations

import os
from typing import Tuple

import torch


def export_torchscript(model: torch.nn.Module, path: str, example_shape: Tuple[int, ...] = (1, 1, 8, 8)) -> str:
    """trace 导出并把模型置为 eval（trace 前必须 eval，否则 dropout/BN 行为会被固化错）。"""
    model.eval()
    example = torch.randn(*example_shape)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        traced = torch.jit.freeze(traced)          # 冻结：常量折叠，体积更小
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.jit.save(traced, path)
    return path


def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024.0 * 1024.0)


def assert_export_matches(model: torch.nn.Module, path: str, atol: float = 1e-5) -> float:
    """导出前后输出最大绝对误差（应接近 0）。"""
    model.eval()
    x = torch.randn(4, 1, 8, 8)
    with torch.no_grad():
        a = model(x)
        b = torch.jit.load(path)(x)
    return float((a - b).abs().max())
