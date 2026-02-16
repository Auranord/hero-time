from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def analyze_video_motion(
    vod_path: str,
    cache_dir: str = "data/cache",
    analysis_fps: float = 2.0,
    window_seconds: int = 30,
    window_overlap_seconds: int = 15,
    scene_change_multiplier: float = 2.5,
) -> dict[str, Any]:
    """Analyze low-FPS motion and scene-change intensity aligned to scoring windows."""

    source_path = Path(vod_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path}")

    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for motion analysis: {source_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = max(analysis_fps, 1.0)
    frame_interval = max(int(round(native_fps / max(analysis_fps, 0.1))), 1)

    samples: list[dict[str, Any]] = []
    prev_gray = None
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            timestamp_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                motion_score = 0.0
            else:
                diff = cv2.absdiff(gray, prev_gray)
                motion_score = float(np.mean(diff) / 255.0)

            samples.append(
                {
                    "sample_index": len(samples),
                    "frame_index": frame_index,
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "motion_score": round(motion_score, 6),
                }
            )
            prev_gray = gray
            frame_index += 1
    finally:
        capture.release()

    motion_values = [sample["motion_score"] for sample in samples]
    motion_mean = float(np.mean(motion_values)) if motion_values else 0.0
    motion_std = float(np.std(motion_values)) if motion_values else 0.0
    scene_change_threshold = motion_mean + (scene_change_multiplier * motion_std)

    motion_peaks = [
        sample for sample in samples if sample["motion_score"] >= max(motion_mean + motion_std, 0.01)
    ]
    scene_changes = [sample for sample in samples if sample["motion_score"] >= scene_change_threshold]

    duration_seconds = _estimate_duration_seconds(capture, samples, native_fps)
    windows = _window_video_features(
        samples=samples,
        duration_seconds=duration_seconds,
        window_seconds=window_seconds,
        window_overlap_seconds=window_overlap_seconds,
        scene_change_threshold=scene_change_threshold,
    )

    artifact_dir = Path(cache_dir).expanduser().resolve() / "features" / "video_motion" / source_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "ok",
        "vod_path": str(source_path),
        "cache_dir": str(artifact_dir),
        "analysis_fps": analysis_fps,
        "native_fps": round(native_fps, 3),
        "sample_interval_frames": frame_interval,
        "duration_seconds": round(duration_seconds, 3),
        "sample_count": len(samples),
        "motion_stats": {
            "mean": round(motion_mean, 6),
            "std": round(motion_std, 6),
            "peak_count": len(motion_peaks),
        },
        "scene_change_threshold": round(scene_change_threshold, 6),
        "scene_change_count": len(scene_changes),
        "scene_change_rate": round(len(scene_changes) / duration_seconds, 6) if duration_seconds > 0 else 0.0,
        "motion_peaks": motion_peaks,
        "scene_changes": scene_changes,
        "window_seconds": window_seconds,
        "window_overlap_seconds": window_overlap_seconds,
        "window_features": windows,
    }

    artifact_path = artifact_dir / "video_motion.json"
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        **payload,
        "video_motion_path": str(artifact_path),
    }


def _estimate_duration_seconds(capture: Any, samples: list[dict[str, Any]], native_fps: float) -> float:
    if samples:
        sample_based = samples[-1]["timestamp_seconds"]
    else:
        sample_based = 0.0

    frame_count = float(capture.get(7) or 0.0)  # cv2.CAP_PROP_FRAME_COUNT
    frame_based = frame_count / native_fps if native_fps > 0 else 0.0
    return max(sample_based, frame_based)


def _window_video_features(
    samples: list[dict[str, Any]],
    duration_seconds: float,
    window_seconds: int,
    window_overlap_seconds: int,
    scene_change_threshold: float,
) -> list[dict[str, Any]]:
    if duration_seconds <= 0:
        return []

    step_seconds = max(window_seconds - window_overlap_seconds, 1)
    windows: list[dict[str, Any]] = []
    index = 0
    start = 0.0

    while start < duration_seconds:
        end = min(start + window_seconds, duration_seconds)
        in_window = [s for s in samples if start <= s["timestamp_seconds"] < end]
        motion_values = [s["motion_score"] for s in in_window]
        scene_change_count = sum(1 for s in in_window if s["motion_score"] >= scene_change_threshold)

        windows.append(
            {
                "window_index": index,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "sample_count": len(in_window),
                "motion_mean": round(sum(motion_values) / len(motion_values), 6) if motion_values else 0.0,
                "motion_peak": round(max(motion_values), 6) if motion_values else 0.0,
                "scene_change_count": scene_change_count,
                "scene_change_frequency": round(scene_change_count / (end - start), 6) if end > start else 0.0,
            }
        )

        start += float(step_seconds)
        index += 1

    return windows
