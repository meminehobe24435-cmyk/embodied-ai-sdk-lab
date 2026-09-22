"""INT8 动态量化（torch.ao.quantization.quantize_dynamic）。

要点（也写进 README 局限里）：动态量化只把 **Linear / LSTM** 的权重转成 int8 并在推理时动态量化激活，
**卷积层不会被动态量化** —— 所以以卷积为主的网络加速有限，收益主要体现在**模型体积**上。
"""
from __future__ import annotations

import torch


def dynamic_quantize(model: torch.nn.Module):
    """返回 INT8 动态量化后的模型（复制原模型，不改动传入对象）。"""
    import copy
    qmodel = copy.deepcopy(model).eval()
    return torch.ao.quantization.quantize_dynamic(qmodel, {torch.nn.Linear}, dtype=torch.qint8)


def compare_accuracy(model, qmodel, x: torch.Tensor, y: torch.Tensor):
    """对比量化前后 top-1 一致率（相对 FP32 基线）与各自的准确率。"""
    with torch.no_grad():
        logits = model(x)
        qlogits = qmodel(x)
        base_acc = float((logits.argmax(1) == y).float().mean())
        q_acc = float((qlogits.argmax(1) == y).float().mean())
        agree = float((logits.argmax(1) == qlogits.argmax(1)).float().mean())
    return base_acc, q_acc, agree
