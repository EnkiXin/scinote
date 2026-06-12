"""Video frame extraction helpers shared across all 5 stages.

These functions wrap PyAV and are memory-safe (never decode every frame into RAM).
All frame extraction goes through here so segment boundaries, pixel budgets,
and frame counts stay consistent between Stage 1 (note generation) and Stage 4
(answer generation).
"""
from __future__ import annotations

import os
from typing import Optional

import av
from PIL import Image

MAX_PIXELS = 360 * 420
FRAMES_PER_SEGMENT = 8
NUM_SEGMENTS_PER_VIDEO = 4
TOTAL_FRAMES = NUM_SEGMENTS_PER_VIDEO * FRAMES_PER_SEGMENT  # 32


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds. Returns 0.0 if unreadable."""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        # Prefer container duration; fall back to frames/fps
        if container.duration is not None and container.duration > 0:
            dur = float(container.duration) / av.time_base
        elif stream.frames > 0 and stream.average_rate:
            dur = float(stream.frames) / float(stream.average_rate)
        else:
            dur = 0.0
        container.close()
        return dur
    except Exception:
        return 0.0


def _resize_pillow(img: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    return img.resize(
        (max(28, int(w * scale)), max(28, int(h * scale))),
        Image.BILINEAR,
    )


def extract_segment_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int = FRAMES_PER_SEGMENT,
    max_pixels: int = MAX_PIXELS,
) -> list[Image.Image]:
    """Sample `n_frames` uniformly from the time range [start_sec, end_sec].

    Falls back to whole-video uniform sampling if duration/fps cannot be read.
    Pads with the last frame if the segment is shorter than `n_frames` frames.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    total_frames = stream.frames

    # Compute target frame indices within the segment
    if fps > 0 and total_frames > 0:
        start_idx = int(start_sec * fps)
        end_idx = min(int(end_sec * fps), total_frames)
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        # n_frames uniformly spaced in [start_idx, end_idx)
        target_idx = set(
            start_idx + int(i * (end_idx - start_idx) / n_frames)
            for i in range(n_frames)
        )
    else:
        # Unknown duration → fall back to first-n uniform of whole video
        target_idx = None

    out: list[Image.Image] = []
    try:
        for i, f in enumerate(container.decode(video=0)):
            if target_idx is not None and i not in target_idx:
                if target_idx and i > max(target_idx):
                    break
                continue
            img = _resize_pillow(f.to_image(), max_pixels)
            out.append(img)
            if len(out) >= n_frames:
                break
    finally:
        container.close()

    # Pad with last frame if we got fewer than requested
    while out and len(out) < n_frames:
        out.append(out[-1])
    return out


def extract_frames_at_indices(
    video_path: str,
    indices: list[int],
    max_pixels: int = MAX_PIXELS,
) -> list[Image.Image]:
    """Decode specific frame indices from a video. Used by Stage 4."""
    target_idx = set(indices)
    indexed: dict[int, Image.Image] = {}
    container = av.open(video_path)
    try:
        for i, f in enumerate(container.decode(video=0)):
            if i in target_idx:
                indexed[i] = _resize_pillow(f.to_image(), max_pixels)
                if len(indexed) >= len(target_idx):
                    break
    finally:
        container.close()
    return [indexed[i] for i in indices if i in indexed]


def segment_time_ranges(duration_sec: float, num_segments: int = NUM_SEGMENTS_PER_VIDEO) -> list[tuple[float, float]]:
    """Divide [0, duration_sec] into `num_segments` equal time ranges."""
    if duration_sec <= 0:
        return [(0.0, 0.0)] * num_segments
    step = duration_sec / num_segments
    return [(i * step, (i + 1) * step) for i in range(num_segments)]
