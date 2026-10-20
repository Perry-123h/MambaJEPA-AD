import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_revision_experiments import prepare, mask_fields, SelectiveSSM


class RevisionPipelineTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({
            "step": [0, 1, 1, 2, 108, 144],
            "customer": ["A"] * 6,
            "merchant": ["M"] * 6,
            "age": ["2"] * 6,
            "gender": ["F"] * 6,
            "category": ["food", "food", "travel", "food", "travel", "food"],
            "amount": [1, 2, 900, 3, 4, 5],
            "fraud": [0, 0, 1, 0, 1, 0],
        })

    def test_same_step_and_future_are_excluded(self):
        first = prepare(self.sample(), history_length=12)
        self.assertEqual(int(first["valid"][1].sum()), 1)
        self.assertEqual(int(first["valid"][2].sum()), 1)
        self.assertEqual(int(first["valid"][3].sum()), 3)
        changed = self.sample()
        changed.loc[5, "amount"] = 999999
        second = prepare(changed, history_length=12)
        np.testing.assert_allclose(first["x"][:5], second["x"][:5])
        np.testing.assert_allclose(first["seq"][:5], second["seq"][:5])

    def test_field_mask_zeroes_whole_groups(self):
        data = prepare(self.sample(), history_length=12)
        x = torch.from_numpy(data["x"][:2])
        masked, decisions = mask_fields(x, data["field_groups"], 1.0, torch.Generator().manual_seed(1))
        self.assertTrue(decisions.all())
        self.assertTrue(torch.equal(masked, torch.zeros_like(masked)))

    def test_selective_ssm_ignores_padding(self):
        model = SelectiveSSM()
        events = torch.randn(2, 4, 24)
        valid = torch.tensor([[False, False, True, True], [False, False, True, True]])
        events[1, :2] = 1000
        events[1, 2:] = events[0, 2:]
        with torch.no_grad():
            a = model(events, valid)
        self.assertTrue(torch.allclose(a[0], a[1], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
