# utils.loaderspec.py

from __future__ import annotations

import json, logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union, Literal


BuiltinCollate = Literal["ragged_xy", "identity"]
LoaderClassStr = str  # e.g. "torch.utils.data:DataLoader" or "torch_geometric.loader:DataLoader"
BuilderStr = str      # e.g. "llm_script:make_dataset" or "llm_script:MyDataset"


@dataclass(frozen=True)
class CollateSpec:
    """
    Collate spec for the loader.

    Rules:
      - For torch DataLoader ragged batches, use builtin="ragged_xy" (recommended) or "identity".
      - For PyG DataLoader, set collate=None (PyG handles batchifng).
      - We do NOT allow arbitrary custom collate callables here (by design, to reduce brittleness).
    """
    builtin: BuiltinCollate

    def to_dict(self) -> Dict[str, Any]:
        return {"builtin": self.builtin}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CollateSpec":
        if "builtin" not in d:
            raise ValueError("CollateSpec missing required key 'builtin'.")
        builtin = d["builtin"]
        if builtin not in ("ragged_xy", "identity"):
            raise ValueError(f"Unsupported builtin collate: {builtin!r}")
        return CollateSpec(builtin=builtin)

@dataclass(frozen=True)
class DatasetSpec:
    """
    How to build the dataset.

    builder:
      - Function path: "llm_script:make_dataset"
          expected signature: make_dataset(events, preproc, train: bool, **kwargs) -> Dataset
      - OR class path: "llm_script:MyDataset"
          expected constructor: MyDataset(events, preproc, train: bool, **kwargs)
    """
    builder: BuilderStr
    kwargs: Dict[str, Any] = field(default_factory=dict)
    expects_train_flag: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "builder": self.builder,
            "kwargs": self.kwargs,
            "expects_train_flag": self.expects_train_flag,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DatasetSpec":
        if "builder" not in d:
            raise ValueError("DatasetSpec missing required key 'builder'.")
        return DatasetSpec(
            builder=d["builder"],
            kwargs=dict(d.get("kwargs", {})),
            expects_train_flag=bool(d.get("expects_train_flag", True)),
        )

@dataclass(frozen=True)
class LoaderParams:
    """
    How to build the DataLoader / PyG DataLoader.
    """
    class_path: LoaderClassStr
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False

    # Either CollateSpec(...) or None (None recommended for PyG DataLoader).
    collate: Optional[CollateSpec] = field(default_factory=lambda: CollateSpec("ragged_xy"))

    # Advanced loader kwargs (you should whitelist keys in your harness when applying).
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_path,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate": (self.collate.to_dict() if self.collate is not None else None),
            "extra_kwargs": self.extra_kwargs,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LoaderParams":
        if "class" not in d:
            raise ValueError("LoaderParams missing required key 'class'.")
        collate_raw = d.get("collate", {"builtin": "ragged_xy"})
        collate = None if collate_raw is None else CollateSpec.from_dict(dict(collate_raw))
        return LoaderParams(
            class_path=d["class"],
            batch_size=int(d.get("batch_size", 64)),
            shuffle=bool(d.get("shuffle", True)),
            num_workers=int(d.get("num_workers", 0)),
            pin_memory=bool(d.get("pin_memory", False)),
            collate=collate,
            extra_kwargs=dict(d.get("extra_kwargs", {})),
        )


@dataclass(frozen=True)
class LoaderSpec:
    """
    Versioned, JSON-serialisable description of the dataset + loader wiring.

    This is the exact snapshot you write at training-time and read at evaluation-time.
    """

    schema_version: int = 1
    script_module: str = "llm_script"  # module name used by your _mount_llm_script()
    dataset: DatasetSpec = field(default_factory=lambda: DatasetSpec(builder="utils.llm_io:EventDataset"))
    loader: LoaderParams = field(default_factory=lambda: LoaderParams(class_path="torch.utils.data:DataLoader"))
    eval_overrides: Dict[str, Any] = field(default_factory=lambda: {"shuffle": False})

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported LoaderSpec schema_version={self.schema_version}. Expected 1.")

        if not isinstance(self.script_module, str) or not self.script_module:
            raise ValueError("script_module must be a non-empty string.")

        if not isinstance(self.loader.batch_size, int) or self.loader.batch_size <= 0:
            raise ValueError("loader.batch_size must be a positive integer.")

        if self.loader.num_workers < 0:
            raise ValueError("loader.num_workers must be >= 0.")

        # Very lightweight sanity checks for import path strings.
        for name, path in [("dataset.builder", self.dataset.builder), ("loader.class", self.loader.class_path)]:
            if not isinstance(path, str) or ":" not in path:
                raise ValueError(f"{name} must be of form 'module:symbol', got {path!r}.")

        # PyG recommendation: collate should be None for PyG DataLoader.
        if "torch_geometric" in self.loader.class_path and self.loader.collate is not None:
            # not strictly forbidden, but strongly suggest setting collate=None.
            pass

        if not isinstance(self.eval_overrides, dict):
            raise ValueError("eval_overrides must be a dict.")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "script_module": self.script_module,
            "dataset": self.dataset.to_dict(),
            "loader": self.loader.to_dict(),
            "eval_overrides": self.eval_overrides,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LoaderSpec":
        if "schema_version" not in d:
            raise ValueError("LoaderSpec missing required key 'schema_version'.")
        spec = LoaderSpec(
            schema_version=int(d["schema_version"]),
            script_module=str(d.get("script_module", "llm_script")),
            dataset=DatasetSpec.from_dict(dict(d.get("dataset", {}))),
            loader=LoaderParams.from_dict(dict(d.get("loader", {}))),
            eval_overrides=dict(d.get("eval_overrides", {"shuffle": False})),
        )
        spec.validate()
        return spec

    def to_json(self, path: Union[str, Path], *, indent: int = 2) -> None:
        """
        Write LoaderSpec to a JSON file.
        """
        p = Path(path)
        payload = self.to_dict()
        p.write_text(json.dumps(payload, indent=indent, sort_keys=True), encoding="utf-8")

    @staticmethod
    def from_json(path: Union[str, Path]) -> "LoaderSpec":
        """
        Read LoaderSpec from a JSON file.
        """
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("LoaderSpec JSON root must be an object.")
        return LoaderSpec.from_dict(d)

    def to_json_str(self, *, indent: int = 2) -> str:
        """
        Convenience: get JSON string (useful for logging).
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

def build_spec_from_preproc(pre: Any, *, script_module: str = "llm_script") -> "LoaderSpec":
    """
    Build a LoaderSpec snapshot from a preprocessor object's make_loader_cfg().

    - This is called in Stage 2 during training.
    - It should NOT execute any dataset/dataloader construction itself.
    - It MUST be deterministic given preproc state.

    Expected cfg keys (all optional):
      dataset_builder: "module:symbol"
      dataset_kwargs: dict

      loader_class: "module:symbol"
      batch_size: int
      shuffle: bool
      num_workers: int
      pin_memory: bool
      collate: "ragged_xy" | "identity" | None   (None only for PyG)
      extra_loader_kwargs: dict

      eval_overrides: dict
    """

    cfg: Dict[str, Any] = {}
    if hasattr(pre, "make_loader_cfg") and callable(getattr(pre, "make_loader_cfg")):
        cfg = pre.make_loader_cfg() or {}
        if not isinstance(cfg, dict):
            raise TypeError("preproc.make_loader_cfg() must return a dict.")

    # Defaults
    dataset_builder = cfg.get("dataset_builder", "utils.llm_io:EventDataset")
    dataset_kwargs  = cfg.get("dataset_kwargs", {}) or {}

    loader_class    = cfg.get("loader_class", "torch.utils.data:DataLoader")
    batch_size      = int(cfg.get("batch_size", 64))
    shuffle         = bool(cfg.get("shuffle", True))
    num_workers     = int(cfg.get("num_workers", 0))
    pin_memory      = bool(cfg.get("pin_memory", False))

    collate_choice  = cfg.get("collate", "ragged_xy")  # can be None for PyG
    if collate_choice is None:
        collate_spec = None
    else:
        if collate_choice not in ("ragged_xy", "identity"):
            raise ValueError(
                f"Unsupported collate={collate_choice!r}. "
                "Must be 'ragged_xy', 'identity', or None (PyG only)."
            )
        collate_spec = CollateSpec(collate_choice)

    extra_loader_kwargs = cfg.get("extra_loader_kwargs", {}) or {}
    if not isinstance(extra_loader_kwargs, dict):
        raise TypeError("extra_loader_kwargs must be a dict.")

    eval_overrides = cfg.get("eval_overrides", {"shuffle": False}) or {"shuffle": False}
    if not isinstance(eval_overrides, dict):
        raise TypeError("eval_overrides must be a dict.")

    spec = LoaderSpec(
        schema_version=1,
        script_module=script_module,
        dataset=DatasetSpec(
            builder=dataset_builder,
            kwargs=dict(dataset_kwargs),
            expects_train_flag=True,
        ),
        loader=LoaderParams(
            class_path=loader_class,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate=collate_spec,
            extra_kwargs=dict(extra_loader_kwargs),
        ),
        eval_overrides=dict(eval_overrides),
    )

    return spec

def enforce_pyg_policy(spec: LoaderSpec, *, require_torch_collate: bool = True) -> LoaderSpec:
    eo = dict(spec.eval_overrides or {})
    eo["shuffle"] = False  # FORCE for all loaders

    is_pyg = "torch_geometric" in spec.loader.class_path
    loader = spec.loader

    if is_pyg:
        if loader.collate is not None:
            logging.warning("PyG loader selected but collate=%r; forcing collate=None.", loader.collate)
            loader = replace(loader, collate=None)  # <-- key line

        eo["batch_size"] = 1  # FORCE for PyG eval

    else:
        if require_torch_collate and loader.collate is None:
            raise ValueError("torch DataLoader requires collate to be 'ragged_xy' or 'identity' for ragged events.")

    spec2 = LoaderSpec(
        schema_version=spec.schema_version,
        script_module=spec.script_module,
        dataset=spec.dataset,
        loader=loader,
        eval_overrides=eo,
    )
    spec2.validate()
    return spec2

def write_loaderspec(base: str, spec: "LoaderSpec", script_dir: Union[str, Path]) -> str:
    """
    Write *_loaderspec.json next to artefacts.

    Returns the full path as a string.
    """

    script_dir = Path(script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)

    spec.validate()
    out_path = script_dir / f"{base}_loaderspec.json"
    spec.to_json(out_path)
    return str(out_path)