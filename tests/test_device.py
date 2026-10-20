import torch

from src.common.device import get_device


def _set(monkeypatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_prefers_cuda_over_mps(monkeypatch):
    _set(monkeypatch, cuda=True, mps=True)
    assert get_device().type == "cuda"


def test_uses_mps_when_no_cuda(monkeypatch):
    _set(monkeypatch, cuda=False, mps=True)
    assert get_device().type == "mps"


def test_falls_back_to_cpu(monkeypatch):
    _set(monkeypatch, cuda=False, mps=False)
    assert get_device().type == "cpu"


def test_mps_fallback_env_is_set():
    import os
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
