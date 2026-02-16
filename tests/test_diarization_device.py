from __future__ import annotations

import sys
from types import SimpleNamespace

from src.features.diarization import _resolve_torch_device


class _FakeTorchDevice:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name


class _FakeTorch:
    def __init__(self, cuda_available: bool):
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)

    def device(self, name: str) -> _FakeTorchDevice:
        return _FakeTorchDevice(name)


def test_resolve_torch_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(cuda_available=True))
    assert str(_resolve_torch_device("auto")) == "cuda"


def test_resolve_torch_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(cuda_available=False))
    assert str(_resolve_torch_device("auto")) == "cpu"


def test_resolve_torch_device_respects_explicit_device(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(cuda_available=False))
    assert str(_resolve_torch_device("cuda:1")) == "cuda:1"
