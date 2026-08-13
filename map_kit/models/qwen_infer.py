"""Integrate motion-guided adaptive sampling with Qwen3-VL inference.

Pipeline:
    video.mp4 --> MotionAdaptiveSampler --> selected frames + true timestamps
             --> interleaved [<t seconds>, image, <t seconds>, image, ..., question]
             --> Qwen3-VL (transformers) --> answer text

Message format matches the Qwen3-VL demo, but uses inline `image` content
(PIL.Image) instead of `image_url`, e.g.:

    {"role": "user", "content": [
        {"type": "text",  "text": "<0.0 seconds>"},
        {"type": "image", "image": <PIL.Image>},
        {"type": "text",  "text": "<2.6 seconds>"},
        {"type": "image", "image": <PIL.Image>},
        ...
        {"type": "text",  "text": "<question>"},
    ]}

Timestamps are the TRUE non-uniform seconds of the adaptively selected frames
(route 2), so the VLM knows the frames are not equally spaced in time.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_MOTIONVLM_ROOT = os.path.normpath(os.path.join(_PKG_DIR, ".."))
if _MOTIONVLM_ROOT not in sys.path:
    sys.path.insert(0, _MOTIONVLM_ROOT)

from map_kit import MotionAdaptiveSampler, SampleResult

DEFAULT_QWEN_CKPT = "/path/to/base_models/Qwen3-VL-2B-Instruct"
DEFAULT_COTRACKER_CKPT = "/path/to/basemodels/scaled_offline.pth"


def _fmt_ts(t: float) -> str:
    # Match the training-time timestamp format exactly (qwen-vl-finetune uses
    # f"<{float(t)} seconds>", e.g. "<0.0 seconds>", "<2.6 seconds>").
    return f"<{float(t)} seconds>"


def build_messages(
    frames: np.ndarray,
    timestamps: np.ndarray,
    question: str,
    system_prompt: str | None = None,
    prompt_hint: str | None = None,
    preamble: str | None = None,
) -> list[dict[str, Any]]:
    """Build a Qwen3-VL chat message with interleaved timestamps + frames.

    Args:
        frames: (B, H, W, 3) uint8 RGB.
        timestamps: (B,) seconds (true, non-uniform).
        question: the task/query text appended after all frames.
        system_prompt: optional system message.
        prompt_hint: optional sentence prepended to the question (after frames),
            e.g. the route-2 non-uniform-sampling hint.
        preamble: optional text placed BEFORE all frames (very front of the user
            turn), e.g. the route-1 trajectory-mark legend so the VLM reads how
            to interpret the drawn arrows before seeing the images.

    Returns:
        `messages` list ready for processor.apply_chat_template.
    """
    from PIL import Image

    content: list[dict[str, Any]] = []
    if preamble:
        content.append({"type": "text", "text": preamble})
    for frame, t in zip(frames, timestamps):
        content.append({"type": "text", "text": _fmt_ts(float(t))})
        content.append({"type": "image", "image": Image.fromarray(frame)})

    q = question if not prompt_hint else f"{prompt_hint}\n{question}"
    content.append({"type": "text", "text": q})

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": content})
    return messages


class MotionAdaptiveQwen3VL:
    """Adaptive-sampling front-end + Qwen3-VL back-end, one object."""

    def __init__(
        self,
        qwen_ckpt: str = DEFAULT_QWEN_CKPT,
        cotracker_ckpt: str = DEFAULT_COTRACKER_CKPT,
        device: str | None = None,
        tracking_fps: float | None = 8.0,
        grid_size: int = 30,
        max_track_frames: int | None = 200,
        compensate_camera: bool = True,
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = "flash_attention_2",
        segment_track: bool = False,
        online: bool = False,
        online_checkpoint: str | None = None,
    ):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler = MotionAdaptiveSampler(
            cotracker_ckpt=cotracker_ckpt,
            device=self.device,
            tracking_fps=tracking_fps,
            grid_size=grid_size,
            max_track_frames=max_track_frames,
            compensate_camera=compensate_camera,
            segment_track=segment_track,
            online=online,
            online_checkpoint=online_checkpoint,
        )
        dtype = getattr(torch, torch_dtype)
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": self.device}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_ckpt, **load_kwargs
        ).eval()
        self.processor = AutoProcessor.from_pretrained(qwen_ckpt)

    def sample(self, video_path: str, budget: int | None = None) -> SampleResult:
        """Run only the adaptive sampler (budget=None -> fps=1 frame count)."""
        return self.sampler.sample(video_path, budget=budget)

    def generate(
        self,
        video_path: str,
        question: str,
        budget: int | None = None,
        system_prompt: str | None = None,
        add_prompt_hint: bool = True,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        mode: str = "adaptive",
        return_debug: bool = False,
    ) -> str | dict:
        """Adaptively sample `video_path`, then answer `question` with Qwen3-VL.

        Args:
            budget: frames to select. None uses the fps=1 frame count.
            add_prompt_hint: prepend the non-uniform-sampling hint to the query.
            mode: "adaptive" (route 2) or "uniform" (equal-budget baseline).
            return_debug: if True, return a dict with the answer plus the
                SampleResult and the built messages (frames stripped for size).
        """
        import torch

        res = self.sampler.sample(video_path, budget=budget, mode=mode)
        # A uniform baseline is equally spaced -> the non-uniform hint would be
        # misleading, so only attach it in adaptive mode.
        hint = res.prompt_hint if (add_prompt_hint and mode == "adaptive") else None
        messages = build_messages(
            res.frames,
            res.timestamps,
            question,
            system_prompt=system_prompt,
            prompt_hint=hint,
        )

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if return_debug:
            return {
                "answer": answer,
                "selected_timestamps_s": [round(float(t), 3) for t in res.timestamps],
                "selected_indices": res.indices.tolist(),
                "budget": int(res.indices.shape[0]),
                "n_tracking_frames": int(res.track_timestamps.shape[0]),
            }
        return answer


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Motion-adaptive sampling + Qwen3-VL inference"
    )
    ap.add_argument("--video", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--budget", type=int, default=None,
                    help="frames to select; default = fps=1 frame count")
    ap.add_argument("--qwen_ckpt", default=DEFAULT_QWEN_CKPT)
    ap.add_argument("--cotracker_ckpt", default=DEFAULT_COTRACKER_CKPT)
    ap.add_argument("--tracking_fps", type=float, default=8.0)
    ap.add_argument("--grid_size", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--no_prompt_hint", action="store_true")
    ap.add_argument("--no_flash_attn", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    engine = MotionAdaptiveQwen3VL(
        qwen_ckpt=args.qwen_ckpt,
        cotracker_ckpt=args.cotracker_ckpt,
        device=args.device,
        tracking_fps=args.tracking_fps,
        grid_size=args.grid_size,
        attn_implementation=None if args.no_flash_attn else "flash_attention_2",
    )
    result = engine.generate(
        args.video,
        args.question,
        budget=args.budget,
        add_prompt_hint=not args.no_prompt_hint,
        max_new_tokens=args.max_new_tokens,
        return_debug=True,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
