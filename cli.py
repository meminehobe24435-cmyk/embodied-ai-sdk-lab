"""CLI 调试工具 + Visualizer（对应 JD 职责 3）。

用法：
    python cli.py train                训练并导出 FP32 / TorchScript / INT8 三份产物
    python cli.py bench                三种模型对比：延迟 / 吞吐 / 体积 / 精度
    python cli.py infer [--save]       跑一批模拟传感器帧并可视化（保存 PNG）
    python cli.py --help
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdk import (InferenceEngine, RingBuffer, SimulatedFrameSource, dynamic_quantize,   # noqa: E402
                 export_torchscript, model_size_mb, train_tiny_model)
from sdk.model import make_dataset                                                     # noqa: E402

# 注意：torch.jit.save 走 C++ 层，在 Windows 上**不认非 ASCII 路径**
# （本项目目录含中文，直接存会报 "Parent directory ... does not exist"）。
# 所以模型产物统一放到 ASCII 的临时目录；可视化图片用 Python 侧写盘，可以留在仓库内。
import tempfile
ART = os.environ.get("AI_SDK_ARTIFACTS") or os.path.join(tempfile.gettempdir(), "embodied_ai_sdk_artifacts")
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
TS_PATH = os.path.join(ART, "tiny_perception.ts")
PT_PATH = os.path.join(ART, "tiny_perception_state.pt")


def cmd_train(args) -> int:
    os.makedirs(ART, exist_ok=True)
    print("① 训练轻量感知模型（合成任务，CPU 秒级）")
    model, acc = train_tiny_model(epochs=args.epochs)
    print("   测试集准确率 = %.4f" % acc)
    os.makedirs(ART, exist_ok=True)
    torch.save(model.state_dict(), PT_PATH)
    export_torchscript(model, TS_PATH)
    qmodel = dynamic_quantize(model)
    torch.jit.save(torch.jit.trace(qmodel, torch.randn(1, 1, 8, 8)), os.path.join(ART, "tiny_perception_int8.ts"))
    print("   已导出：%s" % ART)
    return 0


def cmd_bench(args) -> int:
    os.makedirs(ART, exist_ok=True)
    print("② Benchmark：PyTorch / TorchScript / INT8 对比")
    model, _ = train_tiny_model(epochs=args.epochs, verbose=False)
    export_torchscript(model, TS_PATH)
    ts_engine = InferenceEngine(torch.jit.load(TS_PATH).eval(), name="torchscript")

    qmodel = dynamic_quantize(model)
    q_path = os.path.join(ART, "tiny_perception_int8.ts")
    torch.jit.save(torch.jit.trace(qmodel, torch.randn(1, 1, 8, 8)), q_path)
    q_engine = InferenceEngine(torch.jit.load(q_path).eval(), name="int8-dynamic")

    fp32_engine = InferenceEngine(model, name="pytorch-fp32")

    x, y = make_dataset(256)
    x_test, y_test = x[128:], y[128:]
    with torch.no_grad():
        base_logits = model(x_test)
        ts_logits = ts_engine.model(x_test)
        q_logits = q_engine.model(x_test)
    base_acc = float((base_logits.argmax(1) == y_test).float().mean())
    ts_acc = float((ts_logits.argmax(1) == y_test).float().mean())
    q_acc = float((q_logits.argmax(1) == y_test).float().mean())
    agree = float((base_logits.argmax(1) == q_logits.argmax(1)).float().mean())

    rows = []
    for engine, size_path in ((fp32_engine, PT_PATH), (ts_engine, TS_PATH), (q_engine, q_path)):
        stats = engine.benchmark(batch_size=args.batch, repeat=args.repeat)
        size = model_size_mb(size_path) if os.path.isfile(size_path) else float("nan")
        rows.append((engine.name, stats["latency_ms"], stats["throughput_sps"], size))

    print("\n   %-16s %12s %14s %12s" % ("模型", "单批延迟(ms)", "吞吐(样本/s)", "体积(MB)"))
    print("   " + "-" * 58)
    for name, latency, tput, size in rows:
        print("   %-16s %12.3f %14.1f %12.3f" % (name, latency, tput, size))
    print("\n   精度：FP32 %.4f ｜ TorchScript %.4f ｜ INT8 %.4f ｜ FP32 与 INT8 预测一致率 %.4f"
          % (base_acc, ts_acc, q_acc, agree))
    print("   说明：动态量化只作用于 Linear，卷积不被量化，所以加速有限、收益主要在体积（详见 README）。")
    return 0


def cmd_infer(args) -> int:
    os.makedirs(ART, exist_ok=True)
    print("③ 推理 + 可视化：从模拟传感器读一批帧，输出预测与置信度")
    source = SimulatedFrameSource(fps=30.0)
    buf = RingBuffer(capacity=8)

    model, _ = train_tiny_model(epochs=args.epochs, verbose=False)
    engine = InferenceEngine(model, name="pytorch-fp32")
    frames, preds = [], []
    for ts, frame in source.stream(args.frames):
        buf.push(ts, frame)
    for ts, frame in buf.latest(args.frames):
        frames.append(frame)
    preds = engine.predict(frames)
    for i, ((name, conf), (ts, _)) in enumerate(zip(preds, buf.latest(args.frames))):
        print("   帧 %2d  t=%.3fs  预测=%-4s 置信度=%.3f" % (i, ts, name, conf))
    print("   缓冲统计：%s" % buf.stats())

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = min(4, len(frames))
        rows_n = (len(frames) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(2.4 * cols, 2.6 * rows_n))
        axes = np.atleast_1d(axes).ravel()
        for ax, frame, (name, conf) in zip(axes, frames, preds):
            ax.imshow(frame, cmap="gray")
            ax.set_title("%s %.2f" % (name, conf), fontsize=9)
            ax.axis("off")
        for ax in axes[len(frames):]:
            ax.axis("off")
        out = os.path.join(DOCS, "inference_visualization.png")
        os.makedirs(DOCS, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print("   已保存可视化：%s" % out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="具身感知 SDK 调试工具")
    parser.add_argument("command", choices=["train", "bench", "infer"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--save", action="store_true", help="可视化结果保存为 PNG")
    args = parser.parse_args()
    return {"train": cmd_train, "bench": cmd_bench, "infer": cmd_infer}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
