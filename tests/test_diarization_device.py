from __future__ import annotations

import sys
from types import SimpleNamespace

from src.features.diarization import _load_pyannote_pipeline, _resolve_torch_device


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


class _PipelineAcceptsUseAuth:
    @staticmethod
    def from_pretrained(model: str, use_auth_token: str | None = None):
        return {"model": model, "token": use_auth_token, "mode": "use_auth_token"}


class _PipelineAcceptsTokenOnly:
    @staticmethod
    def from_pretrained(model: str, token: str | None = None):
        return {"model": model, "token": token, "mode": "token"}


class _PipelineRaisesUnexpectedTypeError:
    @staticmethod
    def from_pretrained(model: str, use_auth_token: str | None = None):
        raise TypeError("some other type error")


def test_resolve_torch_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(cuda_available=True))
    assert str(_resolve_torch_device("auto")) == "cuda"


def test_resolve_torch_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(cuda_available=False))
    assert str(_resolve_torch_device("auto")) == "cpu"


def test_resolve_torch_device_respects_explicit_device(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch(cuda_available=False))
    assert str(_resolve_torch_device("cuda:1")) == "cuda:1"


def test_load_pyannote_pipeline_prefers_use_auth_token_when_supported() -> None:
    pipeline = _load_pyannote_pipeline(
        pipeline_class=_PipelineAcceptsUseAuth,
        pipeline_model="pyannote/speaker-diarization-3.1",
        hf_auth_token="hf_abc",
    )

    assert pipeline["mode"] == "use_auth_token"
    assert pipeline["token"] == "hf_abc"


def test_load_pyannote_pipeline_falls_back_to_token_keyword() -> None:
    def _from_pretrained(model: str, use_auth_token: str | None = None, token: str | None = None):
        if use_auth_token is not None:
            raise TypeError("from_pretrained() got an unexpected keyword argument 'use_auth_token'")
        return {"model": model, "token": token, "mode": "token"}

    class _PipelineCompat:
        from_pretrained = staticmethod(_from_pretrained)

    pipeline = _load_pyannote_pipeline(
        pipeline_class=_PipelineCompat,
        pipeline_model="pyannote/speaker-diarization-3.1",
        hf_auth_token="hf_abc",
    )

    assert pipeline["mode"] == "token"
    assert pipeline["token"] == "hf_abc"


def test_load_pyannote_pipeline_propagates_unrelated_type_errors() -> None:
    try:
        _load_pyannote_pipeline(
            pipeline_class=_PipelineRaisesUnexpectedTypeError,
            pipeline_model="pyannote/speaker-diarization-3.1",
            hf_auth_token="hf_abc",
        )
    except TypeError as exc:
        assert "other type error" in str(exc)
    else:
        raise AssertionError("Expected TypeError to be raised")
