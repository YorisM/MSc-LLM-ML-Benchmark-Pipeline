import os, sys, torch, importlib.util, pickle

def _apply_preproc(preproc, x: torch.Tensor) -> torch.Tensor:
    if callable(preproc):
        return preproc(x)
    if hasattr(preproc, "transform"):
        return preproc.transform(x)
    raise TypeError("Pre-processor is neither callable nor has .transform()")

def _mount_llm_script(model_dir: str) -> None:
    """
    Import the LLM-generated script that lives next to the artefacts and
    register it *also* as sys.modules['__main__'] so that
    __main__.MyPreprocessor can be resolved during unpickling.
    Safe to call more than once per process.
    """
    # find the script_<model>_*.py file
    script_path = next(
        f for f in os.listdir(model_dir)
        if f.startswith("script_") and f.endswith(".py")
    )
    script_path = os.path.join(model_dir, script_path)

    # If we already loaded *this* script, nothing to do
    if "__main__" in sys.modules and getattr(sys.modules["__main__"], "__file__", None) == script_path:
        return

    spec = importlib.util.spec_from_file_location("llm_script", script_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)           # type: ignore[attr-defined]

    sys.modules["llm_script"] = mod        # real name
    sys.modules["__main__"]   = mod        # alias used inside pickle

def _initialize_artefacts(model_path: str):
    model_dir = os.path.dirname(model_path)

    # Make sure MyPreprocessor lives in sys.modules['__main__']
    _mount_llm_script(model_dir)

    # Now unpickle safely
    with open(model_path.replace("_model.pkl", "_preproc.pkl"), "rb") as f:
        preproc = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, preproc