"""Motion-guided adaptive frame sampling (route 2).

Public API:
    from map_kit import MotionAdaptiveSampler
"""
from map_kit.core.pipeline import MotionAdaptiveSampler, SampleResult

__all__ = ["MotionAdaptiveSampler", "SampleResult"]


def __getattr__(name):
    # Lazy: importing qwen_infer pulls in transformers; keep it optional.
    if name in ("MotionAdaptiveQwen3VL", "build_messages"):
        from map_kit.models.qwen_infer import MotionAdaptiveQwen3VL, build_messages
        return {"MotionAdaptiveQwen3VL": MotionAdaptiveQwen3VL,
                "build_messages": build_messages}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
