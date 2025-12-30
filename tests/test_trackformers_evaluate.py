# tests/test_trackformers_evaluate.py
import unittest
import numpy as np
import torch

# import the module under test
import challenges.TRACKFORMERS.evaluate_trackformers as ev

class _AssertCallModel(torch.nn.Module):
    """
    Dummy model that asserts it is called with *batch_x only*,
    and returns per-hit int labels matching the provided y.
    """
    def __init__(self, expect: str, return_style: str):
        super().__init__()
        self.expect = expect          # "list" or "tensor"
        self.return_style = return_style  # "list" | "padded_tensor" | "flat_tensor"
        self.last_input_type = None

    def forward(self, batch_x):
        self.last_input_type = "list" if isinstance(batch_x, list) else "tensor"

        if self.expect == "list":
            assert isinstance(batch_x, list), f"expected list batch_x, got {type(batch_x).__name__}"
            assert all(torch.is_tensor(t) for t in batch_x), "expected list[Tensor]"
            # return labels per-event
            if self.return_style == "list":
                return [torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) for x in batch_x]
            if self.return_style == "flat_tensor":
                flat = torch.cat([torch.zeros(x.shape[0], dtype=torch.int64, device=x.device) for x in batch_x], dim=0)
                return flat
            raise AssertionError("unsupported return_style for list input")

        if self.expect == "tensor":
            assert torch.is_tensor(batch_x), f"expected Tensor batch_x, got {type(batch_x).__name__}"
            # batch_x: (B, N, F) -> return (B, N) int labels
            if self.return_style == "padded_tensor":
                B, N = batch_x.shape[0], batch_x.shape[1]
                return torch.zeros((B, N), dtype=torch.int64, device=batch_x.device)
            raise AssertionError("unsupported return_style for tensor input")

        raise AssertionError("bad expect")


def _patch_metrics_to_trivial():
    # Make scoring trivial & deterministic:
    # - fit accuracy: count matches where label==0 and true!=0 etc; we'll just return (truth_hits, truth_hits)
    def _fit_accuracy(labels, true_tid):
        true_tid = np.asarray(true_tid)
        mask = (true_tid != 0)
        truth_hits = int(mask.sum())
        # treat everything as correct for testing
        return truth_hits, truth_hits

    def _hungarian_accuracy(pred, true):
        return 1.0

    ev._fit_accuracy = _fit_accuracy
    ev._hungarian_accuracy = _hungarian_accuracy


class TestEvaluateTRACKFORMERS(unittest.TestCase):
    def setUp(self):
        _patch_metrics_to_trivial()

    def test_ragged_xy_list_calls_model_with_batch_x_list(self):
        model = _AssertCallModel(expect="list", return_style="list")

        # patch artefact loader
        ev._initialize_artefacts = lambda model_path: (model, None)

        # batch = [(X_i, y_i), ...] where y_i is per-hit track_id
        X1 = torch.randn(3, 4)
        y1 = torch.tensor([1, 2, 0], dtype=torch.int64)
        X2 = torch.randn(5, 4)
        y2 = torch.tensor([3, 3, 1, 0, 2], dtype=torch.int64)

        test_loader = [[(X1, y1), (X2, y2)]]

        metrics = ev.evaluate_TRACKFORMERS("dummy.pkl", test_loader)

        self.assertEqual(model.last_input_type, "list")
        self.assertIn("accuracy", metrics)
        self.assertIn("FitAccuracy", metrics)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["FitAccuracy"], 1.0)

    def test_padded_tensor_pair_calls_model_with_tensor(self):
        model = _AssertCallModel(expect="tensor", return_style="padded_tensor")
        ev._initialize_artefacts = lambda model_path: (model, None)

        # batch = (X, y) with padding
        X = torch.randn(2, 6, 4)  # B=2, N=6
        y = torch.tensor([[1, 2, 0, 0, 0, 0],
                          [3, 3, 1, 0, 0, 0]], dtype=torch.int64)

        test_loader = [(X, y)]
        metrics = ev.evaluate_TRACKFORMERS("dummy.pkl", test_loader)

        self.assertEqual(model.last_input_type, "tensor")
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["FitAccuracy"], 1.0)

    def test_ragged_xy_list_flat_output_splits_by_lengths(self):
        model = _AssertCallModel(expect="list", return_style="flat_tensor")
        ev._initialize_artefacts = lambda model_path: (model, None)

        X1 = torch.randn(2, 4)
        y1 = torch.tensor([1, 2], dtype=torch.int64)
        X2 = torch.randn(4, 4)
        y2 = torch.tensor([3, 0, 1, 2], dtype=torch.int64)

        test_loader = [[(X1, y1), (X2, y2)]]
        metrics = ev.evaluate_TRACKFORMERS("dummy.pkl", test_loader)

        # if splitting failed, you'd usually get shape/type errors before here
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["FitAccuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
