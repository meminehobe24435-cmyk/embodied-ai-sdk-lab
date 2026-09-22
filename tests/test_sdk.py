"""自检：python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk import (InferenceEngine, RingBuffer, SimulatedFrameSource, dynamic_quantize,   # noqa: E402
                 export_torchscript, model_size_mb, train_tiny_model)
from sdk.export import assert_export_matches                                             # noqa: E402
from sdk.model import TinyPerceptionNet, make_dataset                                    # noqa: E402


class TestModelAndData(unittest.TestCase):
    def test_dataset_is_deterministic(self):
        x1, y1 = make_dataset(32)
        x2, y2 = make_dataset(32)
        self.assertTrue(torch.equal(x1, x2) and torch.equal(y1, y2))

    def test_model_learns_synthetic_task(self):
        _, acc = train_tiny_model(epochs=40, verbose=False)
        self.assertGreater(acc, 0.8, "合成任务上准确率应显著高于随机（0.25）")


class TestExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = train_tiny_model(epochs=20, verbose=False)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "m.ts")
        export_torchscript(cls.model, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_torchscript_matches_pytorch(self):
        diff = assert_export_matches(self.model, self.path, atol=1e-5)
        self.assertLess(diff, 1e-5, "导出前后输出应逐元素一致")

    def test_export_file_exists_and_loadable(self):
        self.assertTrue(os.path.isfile(self.path))
        self.assertGreater(model_size_mb(self.path), 0.0)
        engine = InferenceEngine(torch.jit.load(self.path).eval(), name="torchscript")
        out = engine.predict([np.full((8, 8), 200, dtype=np.uint8)])
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0][0], str)


class TestQuantize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = train_tiny_model(epochs=30, verbose=False)
        cls.qmodel = dynamic_quantize(cls.model)

    def test_quantized_agrees_with_fp32(self):
        x, y = make_dataset(128)
        x_test = x[64:]
        with torch.no_grad():
            fp32 = self.model(x_test).argmax(1)
            int8 = self.qmodel(x_test).argmax(1)
        agree = float((fp32 == int8).float().mean())
        self.assertGreater(agree, 0.8, "INT8 与 FP32 预测一致率应较高")

    def test_quantized_model_is_smaller(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "fp.ts")
            qp = os.path.join(tmp, "q.ts")
            export_torchscript(self.model, fp)
            torch.jit.save(torch.jit.trace(self.qmodel, torch.randn(1, 1, 8, 8)), qp)
            self.assertLess(model_size_mb(qp), model_size_mb(fp), "INT8 模型体积应更小")

    def test_original_model_untouched(self):
        self.assertFalse(any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear)
                             for m in self.model.modules()),
                         "dynamic_quantize 不应改动传入的模型")


class TestInferenceApi(unittest.TestCase):
    def setUp(self):
        self.engine = InferenceEngine(TinyPerceptionNet(), name="test")

    def test_preprocess_uint8_and_float(self):
        x = self.engine.preprocess([np.full((8, 8), 255, dtype=np.uint8),
                                    np.full((8, 8), 0.5, dtype=np.float32)])
        self.assertEqual(tuple(x.shape), (2, 1, 8, 8))
        self.assertLessEqual(float(x.max()), 1.0001)
        self.assertGreaterEqual(float(x.min()), 0.0)

    def test_predict_shapes_and_confidence(self):
        out = self.engine.predict([np.zeros((8, 8), dtype=np.float32)] * 3)
        self.assertEqual(len(out), 3)
        for name, conf in out:
            self.assertIn(name, InferenceEngine(TinyPerceptionNet()).class_names)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_benchmark_fields(self):
        stats = self.engine.benchmark(batch_size=4, repeat=3)
        self.assertEqual(stats["batch_size"], 4)
        self.assertGreater(stats["latency_ms"], 0.0)
        self.assertGreater(stats["throughput_sps"], 0.0)


class TestSensors(unittest.TestCase):
    def test_ring_buffer_drops_oldest(self):
        buf = RingBuffer(capacity=3)
        for i in range(5):
            buf.push(float(i), np.full((8, 8), i, dtype=np.float32))
        self.assertEqual(len(buf), 3)
        self.assertEqual(buf.dropped, 2)
        latest = buf.latest(3)
        self.assertEqual(int(latest[0][1][0, 0]), 2)
        self.assertEqual(int(latest[-1][1][0, 0]), 4)

    def test_simulated_source_is_reproducible(self):
        a = [f[1].sum() for f in SimulatedFrameSource().stream(3)]
        b = [f[1].sum() for f in SimulatedFrameSource().stream(3)]
        self.assertEqual(a, b)

    def test_source_works_with_inference(self):
        engine = InferenceEngine(TinyPerceptionNet())
        frames = [frame for _, frame in SimulatedFrameSource().stream(4)]
        self.assertEqual(len(engine.predict(frames)), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
