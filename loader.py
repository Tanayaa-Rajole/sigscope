"""Load IQ and WAV files into a common Signal container.

Supported:
  .iq / .cfile / .cf32 / .bin  raw interleaved I,Q samples
                               (complex64 by default; int16, uint8, float32 pairs also work)
  .wav                         stereo -> I (left) + jQ (right)
                               mono   -> real signal, optionally converted to analytic
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

RAW_EXTENSIONS = {".iq", ".cfile", ".cf32", ".bin"}

# name -> (numpy dtype of each I or Q component, converter to float32)
_RAW_FORMATS = {
    "complex64": np.float32,   # float32 I, float32 Q  (GNU Radio cfile)
    "float32": np.float32,
    "int16": np.int16,         # signed 16-bit I/Q
    "uint8": np.uint8,         # RTL-SDR style, centered at 127.5
}


@dataclass
class Signal:
    """Complex baseband samples plus metadata."""
    samples: np.ndarray                # complex64, 1-D
    fs: float = 1.0                    # sample rate in Hz (1.0 = normalized)
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.fs

    @property
    def power(self) -> float:
        return float(np.mean(np.abs(self.samples) ** 2))


def normalize(x: np.ndarray, remove_dc: bool = False) -> np.ndarray:
    """Scale to unit average power (optionally removing DC offset first)."""
    x = np.asarray(x, dtype=np.complex64)
    if remove_dc:
        x = x - np.mean(x)
    p = np.mean(np.abs(x) ** 2)
    if p == 0:
        return x
    return (x / np.sqrt(p)).astype(np.complex64)


def load_iq(path: str | Path, fmt: str = "complex64", fs: float = 1.0,
            norm: bool = True, remove_dc: bool = False,
            max_samples: int | None = None) -> Signal:
    """Load a raw interleaved I/Q file."""
    if fmt not in _RAW_FORMATS:
        raise ValueError(f"Unknown fmt '{fmt}'. Choose from {sorted(_RAW_FORMATS)}.")
    path = Path(path)
    comp_dtype = _RAW_FORMATS[fmt]
    count = -1 if max_samples is None else 2 * max_samples
    raw = np.fromfile(path, dtype=comp_dtype, count=count)
    if raw.size < 2:
        raise ValueError(f"{path} has no complete I/Q pair.")
    if raw.size % 2:
        raw = raw[:-1]  # drop a dangling half-sample

    if fmt == "uint8":
        raw = (raw.astype(np.float32) - 127.5) / 127.5
    elif fmt == "int16":
        raw = raw.astype(np.float32) / 32768.0
    else:
        raw = raw.astype(np.float32)

    x = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    if norm:
        x = normalize(x, remove_dc=remove_dc)
    return Signal(x, fs=fs, source=str(path), meta={"format": fmt})


def load_wav(path: str | Path, analytic: bool = True, norm: bool = True,
             remove_dc: bool = False) -> Signal:
    """Load a WAV file. Sample rate is read from the header."""
    path = Path(path)
    fs, data = wavfile.read(path)

    # integer PCM -> float in [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        if data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)

    if data.ndim == 1:
        # mono: real signal
        x = hilbert(data) if analytic else data.astype(np.complex64)
        layout = "mono-analytic" if analytic else "mono-real"
    elif data.shape[1] >= 2:
        # stereo (or more): first two channels are I and Q
        x = data[:, 0] + 1j * data[:, 1]
        layout = "stereo-iq"
    else:
        raise ValueError("Unsupported WAV channel layout.")

    x = x.astype(np.complex64)
    if norm:
        x = normalize(x, remove_dc=remove_dc)
    return Signal(x, fs=float(fs), source=str(path), meta={"layout": layout})


def load_signal(path: str | Path, **kwargs) -> Signal:
    """Pick the right loader from the file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".wav":
        return load_wav(path, **{k: v for k, v in kwargs.items()
                                 if k in {"analytic", "norm", "remove_dc"}})
    if ext in RAW_EXTENSIONS:
        return load_iq(path, **{k: v for k, v in kwargs.items()
                                if k in {"fmt", "fs", "norm", "remove_dc", "max_samples"}})
    raise ValueError(f"Unsupported file type '{ext}'. "
                     f"Use .wav or one of {sorted(RAW_EXTENSIONS)}.")
