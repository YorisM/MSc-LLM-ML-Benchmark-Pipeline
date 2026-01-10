# challenges.TRACKFORMERS.utils_trackformers.py

import logging, torch


_INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64}


def _is_pyg_obj(obj) -> bool:
    # Duck-typed PyG Data/Batch
    return hasattr(obj, "to") and callable(getattr(obj, "to")) and (hasattr(obj, "x") or hasattr(obj, "pos"))

def _assert_torch_ragged_batch(batch):
    if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
        raise TypeError(f"TORCH lane requires batch == (Xs, ys). Got {type(batch).__name__}.")

    Xs, ys = batch
    if not isinstance(Xs, list) or not isinstance(ys, list):
        raise TypeError(f"TORCH lane requires (list, list). Got Xs={type(Xs).__name__}, ys={type(ys).__name__}")

    if len(Xs) == 0:
        raise ValueError("TORCH lane: empty batch.")
    if len(Xs) != len(ys):
        raise ValueError(f"TORCH lane: len(Xs) != len(ys): {len(Xs)} vs {len(ys)}")

    for i, (x, y) in enumerate(zip(Xs, ys)):
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise TypeError(f"TORCH lane: Xs[{i}] and ys[{i}] must be tensors.")
        if x.ndim != 2:
            raise ValueError(f"TORCH lane: Xs[{i}] must be [N_i,F]. Got {tuple(x.shape)}")
        if y.ndim != 1:
            raise ValueError(f"TORCH lane: ys[{i}] must be [N_i]. Got {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"TORCH lane: N mismatch at i={i}: {x.shape[0]} vs {y.shape[0]}")
        
def _assert_pyg_batch(batch):
    # Prefer a single PyG Batch/Data object. Allow (G, y) only if you want, but I recommend disallowing it.
    if not _is_pyg_obj(batch):
        raise TypeError(f"PyG lane requires a PyG Data/Batch object. Got {type(batch).__name__}")

    x = getattr(batch, "x", None)
    if not (torch.is_tensor(x) and x.ndim == 2):
        raise ValueError("PyG lane requires batch.x to be a rank-2 tensor [num_nodes, F].")

    y = getattr(batch, "y", None)
    if y is None:
        raise ValueError("PyG lane requires batch.y to exist (truth per node).")
    if not (torch.is_tensor(y) and y.ndim == 1 and y.shape[0] == x.shape[0]):
        raise ValueError(f"PyG lane requires batch.y shape [num_nodes]. Got y={None if y is None else tuple(y.shape)}, x={tuple(x.shape)}")

def detect_and_assert_lane(spec, first_batch):
    """
    Returns a string mode: 'torch_ragged_xy' or 'pyg_batch'.
    Raises with a clear message if the batch doesn't match the selected lane.
    """
    is_pyg = "torch_geometric" in spec.loader.class_path

    if is_pyg:
        _assert_pyg_batch(first_batch)
        mode = "pyg_batch"
        logging.info("BATCH_LANE=%s | x=%s y=%s", mode, tuple(first_batch.x.shape), tuple(first_batch.y.shape))
        return mode

    # torch lane
    _assert_torch_ragged_batch(first_batch)
    Xs, ys = first_batch
    mode = "torch_ragged_xy"
    logging.info(
        "BATCH_LANE=%s | B=%d | x0=%s y0=%s",
        mode, len(Xs), tuple(Xs[0].shape), tuple(ys[0].shape)
    )
    return mode

def assert_label_output_by_lane(mode, batch_in, out, *, allow_noise_label=True):
    if mode == "torch_ragged_xy":
        Xs, _ys = batch_in

        if not isinstance(out, list):
            raise TypeError(f"TORCH lane: model must return list[Tensor]. Got {type(out).__name__}")
        if len(out) != len(Xs):
            raise ValueError(f"TORCH lane: output length {len(out)} != B {len(Xs)}")

        for i, (x, yi) in enumerate(zip(Xs, out)):
            if not torch.is_tensor(yi):
                raise TypeError(f"TORCH lane: out[{i}] must be a Tensor.")
            if yi.ndim != 1:
                raise ValueError(f"TORCH lane: out[{i}] must be 1-D (N_i,). Got {tuple(yi.shape)}")
            if yi.dtype not in _INT_DTYPES:
                raise TypeError(f"TORCH lane: out[{i}] must be integer dtype. Got {yi.dtype}")
            if yi.shape[0] != x.shape[0]:
                raise ValueError(f"TORCH lane: length mismatch at i={i}: {yi.shape[0]} vs {x.shape[0]}")
            if allow_noise_label and yi.numel() and yi.min().item() < -1:
                raise ValueError("TORCH lane: labels contain values < -1. Use -1 for noise/unassigned.")

        return  # ok

    if mode == "pyg_batch":
        G = batch_in
        if not torch.is_tensor(out):
            raise TypeError(f"PyG lane: model must return a Tensor[num_nodes]. Got {type(out).__name__}")
        if out.ndim != 1:
            raise ValueError(f"PyG lane: output must be 1-D [num_nodes]. Got {tuple(out.shape)}")
        if out.dtype not in _INT_DTYPES:
            raise TypeError(f"PyG lane: output must be integer dtype. Got {out.dtype}")
        if out.shape[0] != G.x.shape[0]:
            raise ValueError(f"PyG lane: output length {out.shape[0]} != num_nodes {G.x.shape[0]}")
        if allow_noise_label and out.numel() and out.min().item() < -1:
            raise ValueError("PyG lane: labels contain values < -1. Use -1 for noise/unassigned.")
        
        return  # ok

    raise ValueError(f"Unknown mode: {mode}")

def build_trackformers_model(mode, first_batch, make_model, device):
    # Build model + contract check (predict_labels)
    if mode == "torch_ragged_xy":
        Xs, ys = first_batch
        Xs = [x.to(device) for x in Xs]
        model = make_model(Xs).to(device)

        # Enforce: model must expose predict_labels(batch_x)
        if not hasattr(model, "predict_labels") or not callable(getattr(model, "predict_labels")):
            raise TypeError(
                "Contract error: model must implement predict_labels(batch_x). "
                "forward() may return logits/embeddings; predict_labels() must return integer per-hit labels."
            )

        # Run contract check deterministically and without building autograd graphs
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                out = model.predict_labels(Xs)
        except Exception as e:
            raise RuntimeError("predict_labels() crashed in torch_ragged_xy lane.") from e
        finally:
            if was_training:
                model.train()

    elif mode == "pyg_batch":
        G = first_batch.to(device)
        model = make_model(G).to(device)

        # Enforce: model must expose predict_labels(batch_x)
        if not hasattr(model, "predict_labels") or not callable(getattr(model, "predict_labels")):
            raise TypeError(
                "Contract error: model must implement predict_labels(batch_x). "
                "forward() may return logits/embeddings; predict_labels() must return integer per-hit labels."
            )

        # Run contract check deterministically and without building autograd graphs
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                out = model.predict_labels(G)
        except Exception as e:
            raise RuntimeError("predict_labels() crashed in pyg_batch lane.") from e
        finally:
            if was_training:
                model.train()

    else:
        raise RuntimeError(f"Unknown lane mode: {mode}")
    
    assert_label_output_by_lane(mode, first_batch, out, allow_noise_label=True)
    
    return model