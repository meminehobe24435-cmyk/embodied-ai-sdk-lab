"""具身感知 SDK 最小实现：模型推理 API + 传感器接入 + 导出/量化工具。"""
from .model import TinyPerceptionNet, train_tiny_model
from .export import export_torchscript, model_size_mb
from .quantize import dynamic_quantize
from .inference import InferenceEngine, load_engine
from .sensors import RingBuffer, SimulatedFrameSource

__all__ = [
    "TinyPerceptionNet", "train_tiny_model", "export_torchscript", "model_size_mb",
    "dynamic_quantize", "InferenceEngine", "load_engine", "RingBuffer", "SimulatedFrameSource",
]
