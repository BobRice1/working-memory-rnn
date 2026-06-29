"""Torch device selection helpers for CPU and CUDA execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectedDevice:
    """Selected torch device plus a human-readable runtime description."""

    device: torch.device
    description: str


def select_device(requested: str = "auto") -> SelectedDevice:
    """Resolve requested device string into a concrete torch device.

    Args:
        requested: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        Selected device and text describing whether CUDA or CPU will be used.

    Raises:
        ValueError: If ``requested`` is not a supported device option.
    """
    requested = requested.lower()
    cuda_available = torch.cuda.is_available()

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    if requested == "cpu":
        return SelectedDevice(torch.device("cpu"), "Using CPU by request")

    if requested in {"auto", "cuda"} and cuda_available:
        name = torch.cuda.get_device_name(0)
        return SelectedDevice(torch.device("cuda"), f"Using CUDA GPU: {name}")

    if requested == "cuda":
        return SelectedDevice(torch.device("cpu"), "CUDA requested but unavailable; falling back to CPU")

    return SelectedDevice(torch.device("cpu"), "CUDA unavailable; using CPU")
