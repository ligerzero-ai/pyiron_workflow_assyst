"""Post-processing on lists of EngineOutput."""

__all__ = ["collect_relaxation_frames", "export_training_set"]

_DEFERRED = {
    "collect_relaxation_frames": "collect",
    "export_training_set": "export",
}


def __getattr__(name: str):
    if name in _DEFERRED:
        import importlib

        mod = importlib.import_module(f".{_DEFERRED[name]}", __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
