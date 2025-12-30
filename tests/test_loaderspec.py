# tests/test_loaderspec.py

import tempfile
import unittest
from pathlib import Path

from utils.loaderspec import CollateSpec, DatasetSpec, LoaderParams, LoaderSpec, write_loaderspec

# How to run:
# python -m unittest -v tests.test_loaderspec

class TestLoaderSpec(unittest.TestCase):
    @staticmethod
    def example_pyg_spec() -> LoaderSpec:
        return LoaderSpec(
            schema_version=1,
            script_module="llm_script",
            dataset=DatasetSpec(builder="utils.llm_io:EventDataset"),
            loader=LoaderParams(
                class_path="torch_geometric.loader:DataLoader",
                batch_size=4,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate=None,  # crucial for PyG
            ),
            eval_overrides={"shuffle": False, "batch_size": 1},
        )

    @staticmethod
    def example_ragged_spec() -> LoaderSpec:
        return LoaderSpec(
            schema_version=1,
            script_module="llm_script",
            dataset=DatasetSpec(builder="utils.llm_io:EventDataset"),
            loader=LoaderParams(
                class_path="torch.utils.data:DataLoader",
                batch_size=32,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                collate=CollateSpec("ragged_xy"),
                extra_kwargs={"persistent_workers": True},
            ),
            eval_overrides={"shuffle": False},
        )

    # --- Serialization / versioning -------------------------------------------------

    def test_loaderspec_roundtrip_dict(self):
        s1 = self.example_ragged_spec()
        d = s1.to_dict()
        s2 = LoaderSpec.from_dict(d)
        self.assertEqual(s1, s2)

    def test_loaderspec_roundtrip_json(self):
        s1 = self.example_pyg_spec()
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "loaderspec.json"
            s1.to_json(fp)
            s2 = LoaderSpec.from_json(fp)
        self.assertEqual(s1, s2)

    def test_loaderspec_rejects_bad_version(self):
        d = self.example_ragged_spec().to_dict()
        d["schema_version"] = 999
        with self.assertRaises(ValueError) as ctx:
            LoaderSpec.from_dict(d)
        self.assertIn("schema_version", str(ctx.exception))

    # --- Collate policy -------------------------------------------------------------

    def test_collate_rejects_unknown_builtin(self):
        with self.assertRaises(ValueError) as ctx:
            CollateSpec.from_dict({"builtin": "weird"})
        self.assertIn("builtin", str(ctx.exception).lower())

    def test_default_dataset_builder_is_harness_owned(self):
        # This is the key “default is safe” behaviour: LLMs must opt-in to custom dataset explicitly.
        s = LoaderSpec()
        self.assertEqual(s.dataset.builder, "utils.llm_io:EventDataset")

    # --- Optional tests: only run if you expose these helpers -----------------------

    def test_enforce_pyg_policy_if_available(self):
        """
        If you expose enforce_pyg_policy(spec) (in loaderspec.py or llm_io.py),
        ensure it enforces: shuffle False always, and PyG eval batch_size=1 + collate None.
        """
        enforce = None
        try:
            from utils.loaderspec import enforce_pyg_policy as enforce  # type: ignore
        except Exception:
            try:
                from utils.llm_io import enforce_pyg_policy as enforce  # type: ignore
            except Exception:
                enforce = None

        if enforce is None:
            self.skipTest("enforce_pyg_policy not available; skipping policy test")

        spec = self.example_pyg_spec()
        spec2 = enforce(spec)

        # should keep PyG collate None
        self.assertIsNone(spec2.loader.collate)

        # should force eval shuffle False
        eo = dict(spec2.eval_overrides or {})
        self.assertFalse(eo.get("shuffle", True))

        # should force eval batch_size 1 for PyG
        self.assertEqual(eo.get("batch_size", None), 1)

    def test_write_loaderspec_naming_if_available(self):
        """
        If you expose write_loaderspec(base, spec, script_dir), ensure it writes {base}_loaderspec.json.
        """
        try:
            from utils.loaderspec import write_loaderspec  # type: ignore
        except Exception:
            self.skipTest("write_loaderspec not available; skipping naming test")

        with tempfile.TemporaryDirectory() as td:
            spec = self.example_ragged_spec()
            out = write_loaderspec("my_run", spec, td)
            outp = Path(out)
            self.assertTrue(outp.exists())
            self.assertEqual(outp.name, "my_run_loaderspec.json")

            # sanity: it should be valid JSON and load back
            s2 = LoaderSpec.from_json(outp)
            self.assertEqual(spec, s2)


if __name__ == "__main__":
    unittest.main()
