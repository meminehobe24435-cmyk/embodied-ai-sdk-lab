# embodied-ai-sdk-lab · 具身感知 SDK 最小实现

面向**具身智能 SDK**岗位的三个核心工作的最小可运行实现（CPU 就能跑、有实测数字）：

1. **SDK 功能模块与 API 封装**：统一推理 API（模型加载 / 批处理 / 前后处理）+ 传感器接入抽象（环形缓冲 + 模拟帧源），换后端不改业务代码；
2. **模型导出与量化 + Benchmark**：PyTorch → **TorchScript**（含 freeze）→ **INT8 动态量化**，输出**延迟 / 吞吐 / 体积 / 精度**四组对比；
3. **CLI 调试工具 + Visualizer**：`train` / `bench` / `infer` 三个子命令，`infer --save` 输出可视化 PNG。

> 这是为补齐"**SDK 封装 + 模型导出/量化 + Benchmark**"这条能力线自己动手做的练习项目。
> **只用本机已装的 torch / numpy / matplotlib**，不依赖 ONNX / TensorRT（本机没有，见"局限"）。

## 1. 运行

```bash
python cli.py train --epochs 60     # 训练 + 导出 FP32 / TorchScript / INT8
python cli.py bench --batch 8       # 三模型对比：延迟 / 吞吐 / 体积 / 精度
python cli.py infer --frames 8 --save   # 模拟传感器推理 + 可视化 PNG
python -m unittest discover -s tests -v  # 自检
```

## 2. 结构

```
sdk/model.py       轻量感知网（2 层卷积 + 全局池化 + 2 个 Linear）+ 可复现合成任务
sdk/export.py      TorchScript 导出（trace + freeze）+ 导出一致性校验 + 体积统计
sdk/quantize.py    INT8 动态量化 + 量化前后精度对比
sdk/inference.py   统一推理 API：preprocess / predict / benchmark（换后端不改业务代码）
sdk/sensors.py     传感器接入抽象：RingBuffer（丢最旧保最新）+ SimulatedFrameSource
cli.py             命令行：train / bench / infer（含 Visualizer）
tests/test_sdk.py  自检：数据可复现、导出一致、量化一致率、API 形状、环形缓冲语义
```

## 3. 测试覆盖

| 组 | 覆盖内容 |
|---|---|
| 模型与数据 | 数据集**固定种子可复现**；合成任务上准确率显著高于随机 |
| 导出 | **TorchScript 与 PyTorch 输出逐元素一致**（最大误差 < 1e-5）；导出文件可加载并可推理 |
| 量化 | INT8 与 FP32 **预测一致率**达标；**INT8 体积小于 FP32**；`dynamic_quantize` **不改动传入模型** |
| 推理 API | 前处理兼容 `uint8 (0-255)` 与 `float (0-1)`；批处理输出数量与置信度范围正确；benchmark 字段完整 |
| 传感器 | 环形缓冲**丢最旧保最新**且计数正确；模拟帧源**可复现**；与推理 API 串联可用 |

### 实测数字（本机 CPU 输出，`python cli.py bench --batch 8 --repeat 20`）

| 模型 | 单批延迟 (ms) | 吞吐 (样本/s) | 体积 (MB) |
|---|---|---|---|
| pytorch-fp32（内存中） | 0.241 | 33133 | —（无独立文件） |
| **TorchScript（trace+freeze）** | **0.214** | **37348** | 0.135 |
| **INT8 动态量化** | 0.270 | 29652 | **0.059（体积 −56%）** |

**精度**：FP32 1.0000 ｜ TorchScript 1.0000 ｜ INT8 1.0000 ｜ **FP32 与 INT8 预测一致率 100%**
**结论**：动态量化把体积砍掉一半以上、精度无损，但**延迟没有下降**（略升）—— 因为 `quantize_dynamic` 只量化 Linear，卷积仍走 FP32，详见设计决策 2。

## 4. 设计决策（面试常问）

1. **为什么导出用 trace + freeze？** trace 前必须 `eval()`，否则 dropout/BN 的随机行为会被固化错；freeze 做常量折叠，体积更小、推理更快。**导出必须验证"前后输出一致"**，所以单测里直接断言最大误差 < 1e-5。
2. **动态量化为什么加速有限？** `quantize_dynamic` 只把 **Linear / LSTM** 的权重转 int8、激活在推理时动态量化，**卷积层不参与**；本模型以卷积为主，所以**收益主要体现在模型体积**上。要真正加速卷积需要**静态量化（需要校准集）或量化感知训练**，那属于下一步。
3. **为什么把推理包成 `InferenceEngine`？** SDK 的门面价值就在于"换后端不改业务"：上层只调用 `predict(batch)`，底层是 `state_dict` / TorchScript / INT8 都对调用方透明。
4. **传感器层为什么要环形缓冲？** 实时流是无限长的，缓冲必须**有界**；丢最旧保最新 + 丢弃计数，才能既保证推理拿到最新帧、又能观测到"处理跟不上采集"。
5. **benchmark 为什么要 warmup？** 首轮包含内存分配与算子选择，直接计时会把启动开销算进延迟；warmup 后再测才可比。

## 5. 局限（如实说明）

- **没有用 ONNX / onnxruntime**：本机 pip 装不上（代理不可用），所以只有 TorchScript 路径；
- **没有 TensorRT**：本机 torch 是 **CPU 版**，没有 GPU 环境；
- **没有做静态量化 / QAT**：只有动态量化，卷积层未量化，加速有限；
- **没有 ROS2 / Gazebo / Isaac Sim**：本机无 ROS 环境，传感器是**模拟帧源**而非真实设备；
- **没有集成 YOLO / PointNet**：用的是自建小模型做流程验证（任务本身不是重点，流程与指标才是）；
- **路径限制**：`torch.jit.save` 在 Windows 上不认非 ASCII 路径，所以模型产物写在系统临时目录；可视化图片写在 `docs/`。
- 未做多线程/多进程推理服务、批处理调度与内存池优化。

## 6. 许可

MIT，见 [LICENSE](LICENSE)。
