"""轻量感知模型 + 可复现的合成任务（CPU 秒级训练，便于端到端验证导出/量化流程）。

合成任务：8×8 灰度图，按"亮块所在象限"分 4 类 —— 任务本身简单，
重点不在刷精度，而在于提供**可复现的基线**，用来量化"导出/量化前后精度掉了多少"。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyPerceptionNet(nn.Module):
    """小型感知网络：两层卷积 + 全局池化 + 两个全连接（含 Linear，便于做动态量化）。"""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def make_dataset(samples: int = 512, seed: int = 20260922):
    """生成合成数据集：亮块落在哪个象限 → 类别 0/1/2/3（固定种子，结果可复现）。"""
    g = torch.Generator().manual_seed(seed)
    images = torch.zeros(samples, 1, 8, 8)
    labels = torch.zeros(samples, dtype=torch.long)
    for i in range(samples):
        row = 0 if i % 2 == 0 else 4
        col = 0 if (i // 2) % 2 == 0 else 4
        labels[i] = { (0, 0): 0, (0, 4): 1, (4, 0): 2, (4, 4): 3 }[(row, col)]
        block = torch.rand(4, 4, generator=g) * 0.9 + 2.5      # 亮块 ≈ 3.0
        images[i, 0, row:row + 4, col:col + 4] = block
        images[i] += torch.rand(1, 8, 8, generator=g) * 0.4    # 背景噪声
    perm = torch.randperm(samples, generator=g)
    return images[perm], labels[perm]


def train_tiny_model(epochs: int = 60, samples: int = 512, lr: float = 0.02, verbose: bool = True):
    """训练并返回 (model, 测试集准确率)。"""
    torch.manual_seed(20260922)
    images, labels = make_dataset(samples)
    split = int(samples * 0.8)
    x_train, y_train = images[:split], labels[:split]
    x_test, y_test = images[split:], labels[split:]

    model = TinyPerceptionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 20 == 0:
            print("    epoch %2d  loss=%.4f" % (epoch + 1, float(loss)))
    model.eval()
    with torch.no_grad():
        acc = float((model(x_test).argmax(1) == y_test).float().mean())
    return model, acc
