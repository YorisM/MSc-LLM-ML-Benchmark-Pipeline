# tests/test_batch_adapter.py

# run:
# python -m unittest -v tests.test_batch_adapter

import unittest
import torch
from dataclasses import dataclass
from collections import OrderedDict
from utils.llm_io import normalise_batch, extract_batch_x, extract_xy

class TestBatchAdapter(unittest.TestCase):
    def test_tensor_only(self):
        X = torch.randn(4, 8)
        view = normalise_batch(X)
        self.assertTrue(torch.is_tensor(view.batch_x))
        self.assertIsNone(view.batch_y)
        self.assertEqual(view.meta["mode"], "tensor")

    def test_pair_tuple(self):
        X = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        view = normalise_batch((X, y))
        self.assertTrue(torch.is_tensor(view.batch_x))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(view.meta["mode"], "xy_pair")

    def test_ragged_xy_list(self):
        batch = [(torch.randn(3, 4), torch.randint(0, 10, (3,))),
                 (torch.randn(5, 4), torch.randint(0, 10, (5,)))]
        view = normalise_batch(batch)
        self.assertIsInstance(view.batch_x, list)
        self.assertEqual(len(view.batch_x), 2)
        self.assertEqual(view.meta["mode"], "ragged_xy_list")

    def test_ragged_x_list(self):
        batch = [torch.randn(3, 4), torch.randn(5, 4)]
        view = normalise_batch(batch)
        self.assertIsInstance(view.batch_x, list)
        self.assertIsNone(view.batch_y)
        self.assertEqual(view.meta["mode"], "ragged_x_list")

    def test_dict_batch(self):
        X = torch.randn(2, 7)
        y = torch.randint(0, 2, (2,))
        view = normalise_batch({"inputs": X, "labels": y})
        self.assertTrue(torch.is_tensor(view.batch_x))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(view.meta["mode"], "dict")

    def test_device_move(self):
        X = torch.randn(2, 3)
        y = torch.randint(0, 2, (2,))
        dev = torch.device("cpu")
        bx, by, meta = extract_xy((X, y), device=dev)
        self.assertEqual(bx.device.type, "cpu")
        self.assertEqual(by.device.type, "cpu")

    def test_harness_call_style_batch_x_only(self):
        dev = torch.device("cpu")

        # ragged [(X,y), ...]
        batch = [(torch.randn(3,4), torch.zeros(3, dtype=torch.long)),
                (torch.randn(5,4), torch.ones(5, dtype=torch.long))]
        
        view = normalise_batch(batch, device=dev)

        # model should receive only batch_x
        def dummy_model(batch_x):
            assert isinstance(batch_x, list)
            assert torch.is_tensor(batch_x[0])
            return 0

        dummy_model(view.batch_x)
    
    def test_tensor_in_list_singleton(self):
        batch = [torch.randn(2, 3)]
        view = normalise_batch(batch)
        self.assertTrue(torch.is_tensor(view.batch_x))
        self.assertIsNone(view.batch_y)
        self.assertEqual(view.meta["mode"], "tensor_in_list")

    def test_multi_xy_tuple_len_gt_2(self):
        x1 = torch.randn(4, 5)
        x2 = torch.randn(4, 7)
        y  = torch.randint(0, 3, (4,))
        view = normalise_batch((x1, x2, y))
        self.assertIsInstance(view.batch_x, tuple)
        self.assertEqual(len(view.batch_x), 2)
        self.assertTrue(torch.is_tensor(view.batch_x[0]))
        self.assertTrue(torch.is_tensor(view.batch_x[1]))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(view.meta["mode"], "multi_xy_tuple")
        self.assertEqual(view.meta["arity"], 3)

    def test_ragged_multi_xy_list(self):
        # Each sample is (X1, X2, y)
        b = [
            (torch.randn(3, 4), torch.randn(3, 2), torch.randint(0, 5, (3,))),
            (torch.randn(5, 4), torch.randn(5, 2), torch.randint(0, 5, (5,))),
        ]
        view = normalise_batch(b)
        self.assertIsInstance(view.batch_x, list)
        self.assertIsInstance(view.batch_x[0], tuple)
        self.assertEqual(len(view.batch_x[0]), 2)
        self.assertIsInstance(view.batch_y, list)
        self.assertEqual(len(view.batch_y), 2)
        self.assertEqual(view.meta["mode"], "ragged_multi_xy_list")
        self.assertEqual(view.meta["arity"], 3)

    def test_mapping_batch_ordereddict(self):
        X = torch.randn(2, 7)
        y = torch.randint(0, 2, (2,))
        batch = OrderedDict([("inputs", X), ("labels", y)])
        view = normalise_batch(batch)
        self.assertTrue(torch.is_tensor(view.batch_x))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(view.meta["mode"], "dict")

    def test_attr_object_schema(self):
        @dataclass
        class Obj:
            inputs: torch.Tensor
            labels: torch.Tensor

        X = torch.randn(2, 3)
        y = torch.randint(0, 2, (2,))
        view = normalise_batch(Obj(inputs=X, labels=y))
        self.assertTrue(torch.is_tensor(view.batch_x))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(view.meta["mode"], "attr_object")

    def test_pyg_data_schema(self):
        # Dummy PyG-like object: has .to() and .x but no .batch
        class DummyPyGData:
            def __init__(self):
                self.x = torch.randn(3, 4)
                self.y = torch.randint(0, 2, (3,))
            def to(self, device, non_blocking: bool = True):
                self.x = self.x.to(device)
                self.y = self.y.to(device)
                return self

        dev = torch.device("cpu")
        view = normalise_batch(DummyPyGData(), device=dev)
        self.assertEqual(view.meta["mode"], "pyg_data")
        self.assertTrue(hasattr(view.batch_x, "x"))
        self.assertEqual(view.batch_x.x.device.type, "cpu")

    def test_pyg_batch_schema(self):
        # Dummy PyG-like batch: has .batch attribute
        class DummyPyGBatch:
            def __init__(self):
                self.x = torch.randn(6, 4)
                self.y = torch.randint(0, 2, (6,))
                self.batch = torch.zeros(6, dtype=torch.long)
            def to(self, device, non_blocking: bool = True):
                self.x = self.x.to(device)
                self.y = self.y.to(device)
                self.batch = self.batch.to(device)
                return self

        view = normalise_batch(DummyPyGBatch(), device=torch.device("cpu"))
        self.assertEqual(view.meta["mode"], "pyg_batch")
        self.assertTrue(hasattr(view.batch_x, "batch"))

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            _ = normalise_batch(12345)
    
    def test_fourtops_collated_list_xy(self):
        """
        FOURTOPS failure mode: some collate paths can yield a *list of tensors*:
            batch == [X_batch, y_batch]
        normalise_batch should interpret this as (X, y), not ragged-X.
        """
        Xb = torch.randn(8, 92)                 # [B, 92]
        yb = torch.randint(0, 2, (8,))          # [B]
        batch = [Xb, yb]

        view = normalise_batch(batch)
        self.assertTrue(torch.is_tensor(view.batch_x))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(tuple(view.batch_x.shape), (8, 92))
        self.assertEqual(tuple(view.batch_y.shape), (8,))
        self.assertEqual(view, "collated_list_xy")

    def test_fourtops_tuple_of_lists_xy_stacks_to_tensor(self):
        """
        FOURTOPS failure mode: if a model chooses ragged_xy collate by mistake,
        you can see:
            batch == (xs, ys) where xs is list[Tensor(92,)] and ys list[scalar]
        normalise_batch should detect fixed-size samples and stack them.
        """
        B = 6
        xs = [torch.randn(92) for _ in range(B)]                     # each [92]
        ys = [torch.randint(0, 2, ()) for _ in range(B)]             # each scalar tensor
        batch = (xs, ys)

        view = normalise_batch(batch)

        # Expect stacked tensors for fixed-length samples
        self.assertTrue(torch.is_tensor(view.batch_x), msg=f"batch_x type was {type(view.batch_x)}")
        self.assertTrue(torch.is_tensor(view.batch_y), msg=f"batch_y type was {type(view.batch_y)}")
        self.assertEqual(tuple(view.batch_x.shape), (B, 92))
        self.assertEqual(tuple(view.batch_y.shape), (B,))

        # Mode name may vary depending on your implementation; accept either.
        self.assertIn(view, ("xy_pair", "stacked_xy_pair", "stacked_tuple_xy", "xy_pair_stacked"))

    def test_pyg_collated_list_xy(self):
        # Simulate PyG DataLoader returning [Batch, y]
        class DummyPyGBatch:
            def __init__(self):
                self.x = torch.randn(6, 4)
                self.batch = torch.zeros(6, dtype=torch.long)
            def to(self, device, non_blocking: bool = True):
                self.x = self.x.to(device)
                self.batch = self.batch.to(device)
                return self

        y = torch.randint(0, 2, (6,))
        view = normalise_batch([DummyPyGBatch(), y], device=torch.device("cpu"))

        self.assertTrue(hasattr(view.batch_x, "batch"))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(tuple(view.batch_y.shape), (6,))
        self.assertEqual(view.meta["mode"], "pyg_collated_list_xy")

    def test_pyg_ragged_xy_list_batched(self):
        # Requires real torch_geometric Batch.from_data_list
        try:
            from torch_geometric.data import Data
        except Exception:
            self.skipTest("torch_geometric not available")

        d1 = Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        d2 = Data(x=torch.randn(5, 4), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long))
        y1 = torch.tensor(1)
        y2 = torch.tensor(0)

        view = normalise_batch([(d1, y1), (d2, y2)], device=torch.device("cpu"))

        # We should get a PyG Batch object, not a python list
        self.assertTrue(hasattr(view.batch_x, "batch"))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(tuple(view.batch_y.shape), (2,))
        self.assertIn(view.meta["mode"], ("pyg_ragged_xy_list_batched", "pyg_ragged_xy_list"))

    def test_pyg_data_list_batched(self):
        try:
            from torch_geometric.data import Data
        except Exception:
            self.skipTest("torch_geometric not available")

        d1 = Data(x=torch.randn(3, 4), y=torch.tensor([1]), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        d2 = Data(x=torch.randn(4, 4), y=torch.tensor([0]), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))

        view = normalise_batch([d1, d2], device=torch.device("cpu"))

        self.assertTrue(hasattr(view.batch_x, "batch"))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(tuple(view.batch_y.shape), (2,))
        self.assertIn(view.meta["mode"], ("pyg_data_list_batched", "pyg_data_list"))

    def test_collated_list_struct_xy_dict(self):
        B = 4
        x_struct = {
            "g": torch.randn(B, 2),
            "obj": torch.randn(B, 18, 5),
        }
        y = torch.randint(0, 2, (B,), dtype=torch.int64)

        view = normalise_batch([x_struct, y], device=torch.device("cpu"))

        self.assertEqual(view, "collated_list_xy")
        self.assertIsInstance(view.batch_x, dict)
        self.assertTrue(torch.is_tensor(view.batch_x["g"]))
        self.assertTrue(torch.is_tensor(view.batch_x["obj"]))
        self.assertTrue(torch.is_tensor(view.batch_y))
        self.assertEqual(tuple(view.batch_y.shape), (B,))
        self.assertTrue(view.meta.get("structured", False))

        def test_collated_list_struct_xy_tuple(self):
            # Another structured X: a tuple of batched tensors (multi-input packed),
            # still wrapped in [X, y] list.
            B = 5
            x1 = torch.randn(B, 10)
            x2 = torch.randn(B, 3)
            y = torch.randint(0, 2, (B,), dtype=torch.int64)

            view = normalise_batch([(x1, x2), y], device=torch.device("cpu"))

            self.assertEqual(view, "collated_list_xy")
            self.assertIsInstance(view.batch_x, tuple)
            self.assertEqual(len(view.batch_x), 2)
            self.assertTrue(torch.is_tensor(view.batch_y))
            self.assertEqual(tuple(view.batch_y.shape), (B,))
            self.assertTrue(view.meta.get("structured", False))

if __name__ == "__main__":
    unittest.main()