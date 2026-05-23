"""llm_client.py — unified Qwen-VL-72B client for v6.

The v6 plan §3.5 specifies a single Qwen-VL-72B vLLM serve used across
all roles (planner / per-frame OCR / visual_inspect / is_sufficient /
final answer). This module wraps the existing `protonote.cli.VLMClient`
(direct transformers loader) so we can keep the architecture loosely
coupled and swap to a vLLM HTTP backend later if needed.

Public API:
    client = QwenVL72BClient.get_or_create(...)
    text = client.generate_text(prompt, max_tokens=...)
    text = client.generate_image(prompt, frame_pil, max_tokens=...)
    text = client.generate_video(prompt, frames_list, max_tokens=...)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

# Defer heavy imports to method calls so unit tests can mock without GPU
_CLIENT_LOCK = threading.Lock()
_SHARED_CLIENT = None


@dataclass
class QwenVL72BClient:
    """Wrapper around protonote.cli.VLMClient with v6-style methods.

    Use `QwenVL72BClient.get_or_create()` to obtain a process-shared
    instance (so the 72B weights are only loaded once across tools).
    """
    model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    device: str = "auto"     # auto = tensor parallel across visible GPUs

    _impl: any = None

    def __post_init__(self):
        if self._impl is None:
            from protonote.cli import VLMClient
            self._impl = VLMClient(model_name=self.model_name,
                                       device=self.device)

    @classmethod
    def get_or_create(cls, model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
                        device: str = "auto") -> "QwenVL72BClient":
        global _SHARED_CLIENT
        with _CLIENT_LOCK:
            if _SHARED_CLIENT is None:
                _SHARED_CLIENT = cls(model_name=model_name, device=device)
            return _SHARED_CLIENT

    # ── thin wrappers around _impl.generate ─────────────────────────────────

    def generate_text(self, prompt: str, *, system: str = "",
                          max_tokens: int = 200, temperature: float = 0.0
                          ) -> str:
        """Text-only generation. No vision input."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user",
                       "content": [{"type": "text", "text": prompt}]})
        return self._impl.generate(msgs, max_new_tokens=max_tokens)

    def generate_image(self, prompt: str, frame_pil, *, system: str = "",
                           max_tokens: int = 300, temperature: float = 0.0
                           ) -> str:
        """Single image + text prompt."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": [
            {"type": "image", "image": frame_pil},
            {"type": "text", "text": prompt},
        ]})
        return self._impl.generate(msgs, max_new_tokens=max_tokens)

    def generate_video(self, prompt: str, frames: list, *, system: str = "",
                           max_tokens: int = 300, temperature: float = 0.0
                           ) -> str:
        """Multi-frame video + text prompt."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt})
        msgs.append({"role": "user", "content": content})
        return self._impl.generate(msgs, max_new_tokens=max_tokens)
