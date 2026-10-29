"""Checks for the PCG-NDT architecture and deployment interfaces."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pcg_ndt import build_model, load_bundle, parameter_breakdown, save_bundle
from pcg_ndt.dtfe import DTFEEncoder


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        cls.model = build_model().eval()
        rng = np.random.default_rng(41)
        stft = rng.normal(-40, 3, (2, 256, 368))
        stft[0, 120:125] += 10
        iq = rng.normal(size=(2, 12000)) + 1j * rng.normal(size=(2, 12000))
        pulse = iq.copy()
        pulse[:, 400:405] += 20
        cls.prepared = cls.model.prepare_inputs(stft, pulse, iq, progress_every=0)
        cls.batch = cls.prepared.tensor_batch(0, 2, torch.device("cpu"))

    def test_architecture_contract(self):
        model = self.model
        self.assertEqual(parameter_breakdown(model)["total"], 5363575)
        self.assertEqual(sum(isinstance(m, DTFEEncoder) for m in model.modules()), 4)
        self.assertEqual(model.node1.threshold_net[1].in_features, 10)
        self.assertEqual(model.node1.dp_expert.ridge.embedding[0].in_features, 8)
        self.assertEqual(model.node2.direct_concept_encoder[0].in_features, 5)
        self.assertEqual(model.node3.direct_concept_encoder[0].in_features, 3)
        self.assertEqual(model.node2.neural_head[0].in_features, 672)
        self.assertEqual(model.node3.neural_head[0].in_features, 672)

    def test_probabilistic_tree(self):
        with torch.inference_mode():
            output = self.model(self.batch)
        p1, p2, p3 = output["p1"], output["p2"], output["p3"]
        a, b, c = p1.double(), p2.double(), p3.double()
        expected = torch.stack(
            [a * (1 - b), (1 - a) * c, (1 - a) * (1 - c), a * b], 1
        ).float()
        torch.testing.assert_close(
            output["class_probability"], expected, rtol=0, atol=0
        )
        torch.testing.assert_close(expected.sum(1), torch.ones(2), rtol=0, atol=1e-6)
        self.assertTrue(torch.equal(output["prediction"], expected.argmax(1)))

    def test_one_sided_correction(self):
        with torch.inference_mode():
            root = self.model.node1(
                self.batch["stft"], self.batch["node1_physics_probability"]
            )
        base_logit = torch.logit(root["base_probability"]).clamp(-10, 10)
        self.assertTrue(torch.all(root["rescue_evidence"] >= 0))
        self.assertTrue(torch.all(root["logit"] >= base_logit))

    def test_bundle_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_bundle(self.model, path)
            restored = load_bundle(path)
            with torch.inference_mode():
                expected = self.model(self.batch)
                actual = restored(self.batch)
            torch.testing.assert_close(
                actual["class_probability"],
                expected["class_probability"],
                rtol=0,
                atol=0,
            )

    def test_reject_wrong_concept_shape(self):
        scalers = {
            "node1": dict(center=[0.0] * 5, scale=[1.0] * 5),
            "node2": dict(center=[0.0] * 5, scale=[1.0] * 5),
            "node3": dict(center=[0.0] * 3, scale=[1.0] * 3),
        }
        with self.assertRaises(ValueError):
            build_model(scalers=scalers)


if __name__ == "__main__":
    unittest.main()
