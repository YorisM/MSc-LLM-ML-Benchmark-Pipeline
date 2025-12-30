
# tests/test_load_trackformers_test.py
import gzip
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch


class DummyPreproc:
    def __init__(self):
        self.transform_calls = 0

    def transform(self, x):
        self.transform_calls += 1
        return x  # identity

    def make_loader_cfg(self):
        # default: let evaluator pick DataLoader
        return {}

class DummyPreprocWithCfg(DummyPreproc):
    def make_loader_cfg(self):
        return {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            # "loader_class": "torch.utils.data.DataLoader"  # optional
        }

def _write_gz_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        pickle.dump(obj, fh)

def _make_evt(n=5):
    import numpy as np
    return {
        "hit_r": np.random.rand(n).astype("float32"),
        "hit_theta": np.random.rand(n).astype("float32"),
        "hit_z": np.random.rand(n).astype("float32"),
        "layer_id": np.random.randint(1, 10, size=n).astype("int32"),
        "track_id": np.random.randint(0, 4, size=n).astype("int32"),
    }

class TestLoadTrackformersTest(unittest.TestCase):
    def setUp(self):
        # Import here so your project module path is correct in your env.
        # Adjust this import if your evaluator module lives elsewhere.
        from challenges.TRACKFORMERS import evaluate_trackformers as ev
        self.ev = ev

    def test_builds_loader_and_does_not_preproc_whole_events_list(self):
        """
        Ensures we *don't* call _apply_preproc(preproc, events_list) in the loader builder.
        Instead, preproc.transform is called per __getitem__ access.
        """
        tag = "REDVID_10_50_linear_frac0.05"
        events = [_make_evt(3), _make_evt(4), _make_evt(2)]

        with tempfile.TemporaryDirectory() as td:
            # Build a fake evaluator folder layout: <tmp>/evaluate_trackformers.py next to data/test
            fake_eval_dir = Path(td) / "TRACKFORMERS"
            test_file = fake_eval_dir / "data" / "test" / f"{tag}_test.pkl.gz"
            _write_gz_pickle(test_file, {"events": events})

            pre = DummyPreprocWithCfg()

            # Patch:
            # - _initialize_artefacts -> returns (model, preproc)
            # - __file__ -> so load_TRACKFORMERS_test resolves test path relative to it
            with mock.patch.object(self.ev, "_initialize_artefacts", return_value=(object(), pre)):
                with mock.patch.object(self.ev, "__file__", str(fake_eval_dir / "evaluate_trackformers.py")):
                    loader = self.ev.load_TRACKFORMERS_test("whatever_model_path.pkl", tag=tag)

            # No transform yet (Dataset hasn't been indexed)
            self.assertEqual(pre.transform_calls, 0)

            # Indexing should call transform per-event
            batch = next(iter(loader))  # default ragged returns list of (X,y)
            # batch is list of length=batch_size
            self.assertIsInstance(batch, list)
            self.assertGreaterEqual(pre.transform_calls, 1)

            # Validate structure: each item is (X, y)
            x0, y0 = batch[0]
            self.assertTrue(torch.is_tensor(x0))
            self.assertTrue(torch.is_tensor(y0))
            self.assertEqual(x0.ndim, 2)  # (N_i, F)
            self.assertEqual(x0.shape[1], 4)  # r, theta, z, lay_norm

    def test_uses_preproc_collate_fn_if_present(self):
        """
        If preproc exposes _collate_fn, load_TRACKFORMERS_test should use it.
        """
        tag = "REDVID_10_50_linear_frac0.05"
        events = [_make_evt(3), _make_evt(4), _make_evt(2)]

        with tempfile.TemporaryDirectory() as td:
            fake_eval_dir = Path(td) / "TRACKFORMERS"
            test_file = fake_eval_dir / "data" / "test" / f"{tag}_test.pkl.gz"
            _write_gz_pickle(test_file, {"events": events})

            pre = DummyPreprocWithCfg()

            # A collate that returns a tuple (batch_x, batch_y) to prove it’s used.
            def _collate_fn(batch):
                xs = [xy[0] for xy in batch]
                ys = [xy[1] for xy in batch]
                return (xs, ys)

            pre._collate_fn = staticmethod(_collate_fn)

            with mock.patch.object(self.ev, "_initialize_artefacts", return_value=(object(), pre)):
                with mock.patch.object(self.ev, "__file__", str(fake_eval_dir / "evaluate_trackformers.py")):
                    loader = self.ev.load_TRACKFORMERS_test("whatever_model_path.pkl", tag=tag)

            out = next(iter(loader))
            self.assertIsInstance(out, tuple)
            self.assertEqual(len(out), 2)
            bx, by = out
            self.assertIsInstance(bx, list)
            self.assertIsInstance(by, list)

    def test_missing_file_raises(self):
        tag = "REDVID_10_50_linear_frac0.05"
        with tempfile.TemporaryDirectory() as td:
            fake_eval_dir = Path(td) / "TRACKFORMERS"
            pre = DummyPreprocWithCfg()
            with mock.patch.object(self.ev, "_initialize_artefacts", return_value=(object(), pre)):
                with mock.patch.object(self.ev, "__file__", str(fake_eval_dir / "evaluate_trackformers.py")):
                    with self.assertRaises(FileNotFoundError):
                        _ = self.ev.load_TRACKFORMERS_test("whatever_model_path.pkl", tag=tag)

if __name__ == "__main__":
    unittest.main()
