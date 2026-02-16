from __future__ import annotations


def analyze_video_motion(vod_path: str) -> dict:
    """Video intensity proxy contract (motion / scene changes)."""

    return {
        "vod_path": vod_path,
        "motion_peaks": [],
        "scene_change_rate": 0.0,
        "status": "not_implemented",
    }
