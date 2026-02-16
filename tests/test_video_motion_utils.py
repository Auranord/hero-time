from __future__ import annotations

import numpy as np

from src.features.video_motion import _resize_for_motion


class _FakeCv2:
    INTER_AREA = 3

    def __init__(self) -> None:
        self.called_with: tuple[tuple[int, int], int] | None = None

    def resize(self, frame, dims, interpolation):
        self.called_with = (dims, interpolation)
        target_w, target_h = dims
        return np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)


def test_resize_for_motion_skips_when_width_already_small() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    fake_cv2 = _FakeCv2()

    resized = _resize_for_motion(frame=frame, processing_width=320, cv2_module=fake_cv2)

    assert resized is frame
    assert fake_cv2.called_with is None


def test_resize_for_motion_resizes_down_to_target_width() -> None:
    frame = np.zeros((100, 800, 3), dtype=np.uint8)
    fake_cv2 = _FakeCv2()

    resized = _resize_for_motion(frame=frame, processing_width=320, cv2_module=fake_cv2)

    assert resized.shape[1] == 320
    assert fake_cv2.called_with is not None


def test_resize_for_motion_disables_resize_for_non_positive_width() -> None:
    frame = np.zeros((100, 800, 3), dtype=np.uint8)
    fake_cv2 = _FakeCv2()

    resized = _resize_for_motion(frame=frame, processing_width=0, cv2_module=fake_cv2)

    assert resized is frame
    assert fake_cv2.called_with is None
