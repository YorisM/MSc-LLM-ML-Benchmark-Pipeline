# tests/test_llm_io_workflow.py
import unittest
from unittest.mock import patch
import numpy as np
import torch

from utils.loaderspec import CollateSpec, DatasetSpec, LoaderParams, LoaderSpec
from utils.llm_io import assert_label_output, build_dataset, build_dataloader, split_X_y, EventDataset


# How to run:
# python -m unittest -v tests.test_llm_io_workflow


class DummyPreproc:
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0

class DummyLoader:
    """Captures kwargs passed by build_dataloader."""
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs

class DummyData:
    """PyG-like Data object for length inference."""
    def __init__(self, n: int):
        self.x = torch.randn(n, 3)
        self.num_nodes = n

class TestLLMIoWorkflow(unittest.TestCase):
    def test_split_X_y_shapes_and_dtypes(self):
        evt = {
            "hit_r":     np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "hit_theta": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "hit_z":     np.array([10.0, 11.0, 12.0], dtype=np.float32),
            "layer_id":  np.array([2, 3, 4], dtype=np.int32),
            "track_id":  np.array([5, 5, 9], dtype=np.int64),
        }
        X, y = split_X_y(evt)
        self.assertEqual(tuple(X.shape), (3, 4))
        self.assertEqual(tuple(y.shape), (3,))
        self.assertEqual(X.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.int64)

    def test_eventdataset_applies_preproc(self):
        events = [{
            "hit_r":     np.array([1.0], dtype=np.float32),
            "hit_theta": np.array([0.1], dtype=np.float32),
            "hit_z":     np.array([10.0], dtype=np.float32),
            "layer_id":  np.array([2], dtype=np.int32),
            "track_id":  np.array([7], dtype=np.int64),
        }]
        ds = EventDataset(events, DummyPreproc(), train=True)
        X, y = ds[0]
        # original X first column would be 1.0 -> after +1 => 2.0
        self.assertAlmostEqual(float(X[0, 0].item()), 2.0, places=6)
        self.assertEqual(int(y[0].item()), 7)

    def test_build_dataset_calls_function_builder_with_train_and_kwargs(self):
        called = {}

        def make_ds(events, preproc, *, train: bool, alpha: int = 0):
            called["train"] = train
            called["alpha"] = alpha
            return ("ok", train, alpha)

        spec = LoaderSpec(
            schema_version=1,
            script_module="llm_script",
            dataset=DatasetSpec(builder="X:make_ds", kwargs={"alpha": 42}),
            loader=LoaderParams(class_path="torch.utils.data:DataLoader", batch_size=1),
        )

        with patch("utils.llm_io.resolve_path", side_effect=lambda p: make_ds if p == "X:make_ds" else None):
            ds = build_dataset(spec, events=[{"dummy": True}], preproc=None, train=False)

        self.assertEqual(ds, ("ok", False, 42))
        self.assertEqual(called["train"], False)
        self.assertEqual(called["alpha"], 42)

    def test_build_dataset_calls_class_builder_with_train_and_kwargs(self):
        class MyDS:
            def __init__(self, events, preproc, *, train: bool, beta: int = 0):
                self.train = train
                self.beta = beta

        spec = LoaderSpec(
            schema_version=1,
            script_module="llm_script",
            dataset=DatasetSpec(builder="X:MyDS", kwargs={"beta": 7}),
            loader=LoaderParams(class_path="torch.utils.data:DataLoader", batch_size=1),
        )

        with patch("utils.llm_io.resolve_path", side_effect=lambda p: MyDS if p == "X:MyDS" else None):
            ds = build_dataset(spec, events=[{"dummy": True}], preproc=None, train=True)

        self.assertIsInstance(ds, MyDS)
        self.assertTrue(ds.train)
        self.assertEqual(ds.beta, 7)

    def test_build_dataloader_passes_collate_fn_for_torch_loader(self):
        spec = LoaderSpec(
            schema_version=1,
            script_module="llm_script",
            dataset=DatasetSpec(builder="utils.llm_io:EventDataset"),
            loader=LoaderParams(
                class_path="torch.utils.data:DataLoader",
                batch_size=16,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate=CollateSpec("identity"),
                extra_kwargs={"drop_last": True},
            ),
            eval_overrides={"shuffle": False},
        )

        ds = [1, 2, 3]  # dummy dataset

        # Patch resolve_path to avoid importing real DataLoader
        with patch("utils.llm_io.resolve_path", return_value=DummyLoader):
            loader = build_dataloader(spec, ds, is_eval=True)

        # collate_fn should be present for torch DataLoader
        self.assertIn("collate_fn", loader.kwargs)
        self.assertFalse(loader.kwargs["shuffle"])  # eval override applied
        self.assertTrue(loader.kwargs["drop_last"])  # extra_kwargs forwarded

    def test_build_dataloader_pyg_does_not_pass_collate_fn(self):
        # We avoid torch_geometric import by patching resolve_path.
        spec = LoaderSpec(
            schema_version=1,
            script_module="llm_script",
            dataset=DatasetSpec(builder="utils.llm_io:EventDataset"),
            loader=LoaderParams(
                class_path="torch_geometric.loader:DataLoader",
                batch_size=8,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate=None,
            ),
            eval_overrides={"batch_size": 1, "shuffle": False},
        )
        ds = [1, 2, 3]

        with patch("utils.llm_io.resolve_path", return_value=DummyLoader):
            loader = build_dataloader(spec, ds, is_eval=True)

        self.assertNotIn("collate_fn", loader.kwargs)
        self.assertEqual(loader.kwargs["batch_size"], 1)
        self.assertFalse(loader.kwargs["shuffle"])

    # -------- assert_label_output tests --------

    def test_assert_label_output_accepts_list_per_event(self):
        batch_x = [torch.randn(3, 4), torch.randn(5, 4)]
        out = [torch.zeros(3, dtype=torch.long), torch.ones(5, dtype=torch.long)]
        assert_label_output(batch_x, out, allow_noise_label=True)  # should not raise

    def test_assert_label_output_accepts_concatenated_tensor(self):
        batch_x = [torch.randn(3, 4), torch.randn(5, 4)]
        out = torch.zeros(8, dtype=torch.long)
        assert_label_output(batch_x, out, allow_noise_label=True)  # should not raise

    def test_assert_label_output_accepts_N_by_1_tensor(self):
        batch_x = [torch.randn(4, 4)]
        out = torch.zeros(4, 1, dtype=torch.long)  # (N,1) should be accepted
        assert_label_output(batch_x, out, allow_noise_label=True)

    def test_assert_label_output_rejects_float_tensor(self):
        batch_x = [torch.randn(3, 4)]
        out = torch.randn(3)  # float
        with self.assertRaises(TypeError):
            assert_label_output(batch_x, out)

    def test_assert_label_output_rejects_wrong_shape_tensor(self):
        batch_x = [torch.randn(3, 4)]
        out = torch.zeros(3, 2, dtype=torch.long)  # (N,2) not allowed
        with self.assertRaises(ValueError):
            assert_label_output(batch_x, out)

    def test_assert_label_output_rejects_length_mismatch(self):
        batch_x = [torch.randn(3, 4), torch.randn(5, 4)]
        out = torch.zeros(7, dtype=torch.long)  # should be 8
        with self.assertRaises(ValueError):
            assert_label_output(batch_x, out)

    def test_assert_label_output_allows_minus1_but_not_less(self):
        batch_x = [torch.randn(3, 4)]
        ok = torch.tensor([-1, 0, 1], dtype=torch.long)
        bad = torch.tensor([-2, 0, 1], dtype=torch.long)

        assert_label_output(batch_x, ok, allow_noise_label=True)  # should pass
        with self.assertRaises(ValueError):
            assert_label_output(batch_x, bad, allow_noise_label=True)

    def test_assert_label_output_infers_lengths_from_pyg_like_data(self):
        batch_x = [DummyData(4), DummyData(2)]
        out = [torch.zeros(4, dtype=torch.long), torch.zeros(2, dtype=torch.long)]
        assert_label_output(batch_x, out, allow_noise_label=True)

if __name__ == "__main__":
    unittest.main()