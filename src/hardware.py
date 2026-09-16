"""GPU auto-detection and training defaults."""

from dataclasses import dataclass


@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    cuda_version: str
    available: bool

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024


def detect_gpu() -> GPUInfo:
    """Detect GPU and return its info."""
    try:
        import torch

        if not torch.cuda.is_available():
            return GPUInfo(name="CPU", vram_mb=0, cuda_version="", available=False)

        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        vram_mb = total_bytes // (1024 * 1024)
        cuda_version = torch.version.cuda or ""

        return GPUInfo(name=name, vram_mb=vram_mb, cuda_version=cuda_version, available=True)
    except Exception:
        return GPUInfo(name="Unknown", vram_mb=0, cuda_version="", available=False)


def get_hardware_summary(gpu: GPUInfo | None = None) -> str:
    """Human-readable hardware summary for UI display."""
    if gpu is None:
        gpu = detect_gpu()

    if not gpu.available:
        return "**GPU:** Not available (CPU mode)"

    lines = [
        f"**GPU:** {gpu.name}",
        f"**VRAM:** {gpu.vram_gb:.1f} GB",
        f"**CUDA:** {gpu.cuda_version}",
    ]
    return "\n".join(lines)
