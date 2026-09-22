"""传感器接入抽象：把"数据从哪来"和"模型怎么用"解耦（对应 JD 里的传感器接入 API）。"""
from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, List, Optional, Tuple

import numpy as np


class RingBuffer:
    """定长环形缓冲：丢最旧、保最新，用于实时流式输入（与设备侧固件里的做法一致）。"""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity 必须为正")
        self.capacity = capacity
        self._items: Deque[Tuple[float, np.ndarray]] = deque(maxlen=capacity)
        self.dropped = 0

    def push(self, timestamp: float, frame: np.ndarray) -> None:
        if len(self._items) == self.capacity:
            self.dropped += 1
        self._items.append((float(timestamp), np.asarray(frame)))

    def latest(self, count: int = 1) -> List[Tuple[float, np.ndarray]]:
        return list(self._items)[-count:]

    def __len__(self) -> int:
        return len(self._items)

    def stats(self) -> dict:
        return {"capacity": self.capacity, "size": len(self._items), "dropped": self.dropped}


class SimulatedFrameSource:
    """模拟传感器帧源：固定种子生成与训练分布一致的画面，用于无硬件联调。"""

    def __init__(self, fps: float = 30.0, seed: int = 7):
        self.fps = fps
        self.frame_index = 0
        self._rng = np.random.RandomState(seed)

    def read(self) -> Tuple[float, np.ndarray]:
        row = 0 if (self.frame_index // 2) % 2 == 0 else 4
        col = 0 if self.frame_index % 2 == 0 else 4
        frame = self._rng.rand(8, 8).astype(np.float32) * 0.4
        frame[row:row + 4, col:col + 4] += self._rng.rand(4, 4).astype(np.float32) * 0.9 + 2.5
        ts = self.frame_index / self.fps
        self.frame_index += 1
        return ts, frame

    def stream(self, count: int) -> Iterator[Tuple[float, np.ndarray]]:
        for _ in range(count):
            yield self.read()
