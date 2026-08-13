"""Minimal thread-safe client for an OpenAI-compatible chat/vision endpoint.

OpenAI-compatible /v1/chat/completions. Frames are sent as base64 JPEG data
URLs in `image_url` content parts. Used for evaluating our frame-sampling
methods on a stronger closed model (GPT-5.5) — no local GPU / model needed, the
work is HTTP-bound, so runners parallelize with a thread pool.

The endpoint and API key are read from environment variables so no credentials
or provider-identifying URLs are committed:
    MAP_BASE_URL : full chat-completions URL (OpenAI-compatible)
    MAP_API_KEY  : bearer token for that endpoint
    MAP_MODEL    : model name (optional; defaults to GPT-5.5)
"""
from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
import requests

BASE_URL = os.environ.get(
    "MAP_BASE_URL", "https://<your-openai-compatible-endpoint>/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("MAP_MODEL", "GPT-5.5")


def frame_to_data_url(frame: np.ndarray, jpeg_quality: int = 90) -> str:
    """(H,W,3) uint8 RGB -> data:image/jpeg;base64 URL."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_vision_content(frames, timestamps=None, question="",
                         prefix=None, jpeg_quality=90):
    """OpenAI content list: optional prefix, then <t s> text + image per frame,
    then the question. `frames` is a list/array of (H,W,3) uint8 RGB."""
    content = []
    if prefix:
        content.append({"type": "text", "text": prefix})
    for i, fr in enumerate(frames):
        if timestamps is not None:
            content.append({"type": "text",
                            "text": f"<{float(timestamps[i])} seconds>"})
        content.append({"type": "image_url",
                        "image_url": {"url": frame_to_data_url(fr, jpeg_quality)}})
    if question:
        content.append({"type": "text", "text": question})
    return content


class GPTClient:
    """Thread-safe (stateless per call) GPT chat client with retries."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 base_url: str = BASE_URL, timeout: int = 120,
                 max_retries: int = 5):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, content, system=None, max_tokens=64,
             temperature=None) -> str:
        """One vision/text turn. `content` is an OpenAI content list or str.
        Returns the assistant text (empty string on unrecoverable failure).

        temperature defaults to None (omit the field): GPT-5.5 only accepts the
        default temperature=1 and 400s on any explicit value including 0."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens}
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.base_url, headers=headers, json=payload,
                                  timeout=self.timeout)
                if r.status_code == 200:
                    d = r.json()
                    ch = d.get("choices")
                    if ch:
                        msg = ch[0].get("message", {}).get("content")
                        if msg is not None:
                            return msg.strip()
                    last_err = f"no content: {str(d)[:200]}"
                else:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    # Drop an unsupported param and retry immediately.
                    if r.status_code == 400 and "temperature" in payload \
                            and "temperature" in r.text:
                        payload.pop("temperature")
                        continue
                    # Other 4xx (not rate limit) -> unrecoverable, stop.
                    if 400 <= r.status_code < 500 and r.status_code != 429:
                        break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 20))
        print(f"[gpt-warn] failed after {self.max_retries} tries: {last_err}")
        return ""
