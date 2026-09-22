"""推理引擎：对上层提供统一 API（无论底层是 PyTorch / TorchScript / 量化模型）。

这是 SDK 的门面：调用方只看到 load_engine(...).predict(batch)，
不需要知道模型是 .pt、TorchScript 还是 INT8 量化模型 —— 换后端不改业务代码。
"""
from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np
import torch

from .model import TinyPerceptionNet

CLASS_NAMES = ("左上", "右上", "左下", "右下")


class InferenceEngine:
    def __init__(self, model, name: str = "pytorch", class_names: Sequence[str] = CLASS_NAMES):
        self.model = model.eval() if hasattr(model, "eval") else model
        self.name = name
        self.class_names = tuple(class_names)

    # ---------- 前后处理 ----------
    @staticmethod
    def preprocess(images: Sequence[np.ndarray]) -> torch.Tensor:
        """归一化到 [0,1] 并补齐通道维度：接受 (H,W) 或 (H,W,C) 的 uint8/float 数组。"""
        batch: List[np.ndarray] = []
        for img in images:
            arr = np.asarray(img, dtype=np.float32)
            if arr.max() > 1.5:                       # 0-255 的图 → 归一化
                arr = arr / 255.0
            if arr.ndim == 2:
                arr = arr[None, :, :]
            else:
                arr = arr.transpose(2, 0, 1)[:1]      # 只取第一个通道（本模型是单通道）
            batch.append(arr)
        return torch.from_numpy(np.stack(batch))

    def predict(self, images: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        """返回 [(类别名, 置信度)]，与输入一一对应。"""
        x = self.preprocess(images)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return [(self.class_names[int(i)], float(c)) for c, i in zip(conf, idx)]

    # ---------- 性能测量 ----------
    def benchmark(self, batch_size: int, input_shape: Tuple[int, int, int] = (1, 8, 8),
                  warmup: int = 5, repeat: int = 20) -> dict:
        x = torch.randn(batch_size, *input_shape)
        with torch.no_grad():
            for _ in range(warmup):
                self.model(x)
            start = time.perf_counter()
            for _ in range(repeat):
                self.model(x)
            elapsed = time.perf_counter() - start
        return {
            "name": self.name,
            "batch_size": batch_size,
            "total_ms": elapsed * 1000.0,
            "latency_ms": elapsed * 1000.0 / repeat,
            "throughput_sps": batch_size * repeat / elapsed,
        }


def load_engine(path: str, kind: str = "torchscript") -> InferenceEngine:
    """按类型加载：torchscript（.ts）/ pytorch（state_dict 需配合模型结构）。"""
    if kind == "torchscript":
        model = torch.jit.load(path)
        model.eval()
        return InferenceEngine(model, name="torchscript")
    raise ValueError("不支持的模型类型：%s" % kind)


def engine_from_state_dict(state_path: str) -> InferenceEngine:
    model = TinyPerceptionNet()
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    return InferenceEngine(model, name="pytorch")
