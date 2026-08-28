"""GPU/CPU device detection. Never assumes a particular GPU vendor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device_type: str  # "cuda", "mps", or "cpu"
    device_name: str
    torch_device: "object"
    vram_gb: float | None
    supports_amp: bool


def detect_device(prefer: str | None = None) -> DeviceInfo:
    """Detect the best available compute device.

    Parameters
    ----------
    prefer:
        Optional explicit device request ("cuda", "mps", "cpu"). If the
        requested device is unavailable, falls back to CPU with a warning
        logged by the caller.
    """
    import torch

    from enhancoai.utils.logging import get_logger

    logger = get_logger(__name__)

    if prefer == "cpu":
        return DeviceInfo("cpu", "CPU", torch.device("cpu"), None, False)

    if (prefer in (None, "cuda")) and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        return DeviceInfo("cuda", name, torch.device("cuda", idx), round(vram, 2), True)

    if (prefer in (None, "mps")) and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return DeviceInfo("mps", "Apple Silicon (MPS)", torch.device("mps"), None, False)

    if prefer not in (None, "cpu"):
        logger.warning("Requested device '%s' unavailable, falling back to CPU.", prefer)

    return DeviceInfo("cpu", "CPU", torch.device("cpu"), None, False)


def describe(info: DeviceInfo) -> str:
    parts = [f"Device: {info.device_name} ({info.device_type})"]
    if info.vram_gb is not None:
        parts.append(f"VRAM: {info.vram_gb} GB")
    parts.append(f"AMP: {'available' if info.supports_amp else 'unavailable'}")
    return " | ".join(parts)
