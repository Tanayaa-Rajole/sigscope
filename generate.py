"""Synthetic baseband signal generator for SigScope.

All generators return complex64 IQ samples with unit average power
(before noise), so SNR is well defined.

Usage:
    python -m sigscope.generate --mod qpsk --snr 10 --n 4096 --out data/qpsk.iq
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

MODULATIONS = ("am", "fm", "bpsk", "qpsk", "qam16")


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _normalize(x: np.ndarray) -> np.ndarray:
    """Scale to unit average power."""
    p = np.mean(np.abs(x) ** 2)
    return (x / np.sqrt(p)).astype(np.complex64)


def gen_am(n: int, fs: float = 1.0, f_msg: float = 0.01, depth: float = 0.5,
           seed: int | None = None) -> np.ndarray:
    t = np.arange(n) / fs
    msg = np.cos(2 * np.pi * f_msg * t)
    return _normalize((1 + depth * msg).astype(np.complex64))


def gen_fm(n: int, fs: float = 1.0, f_msg: float = 0.01, dev: float = 0.1,
           seed: int | None = None) -> np.ndarray:
    t = np.arange(n) / fs
    # Integral of cos message -> sine; phase = 2*pi*dev/f_msg * sin(...)
    phase = 2 * np.pi * (dev / f_msg) * np.sin(2 * np.pi * f_msg * t)
    return _normalize(np.exp(1j * phase))


def _symbols(n_sym: int, constellation: np.ndarray, seed: int | None) -> np.ndarray:
    idx = _rng(seed).integers(0, len(constellation), n_sym)
    return constellation[idx]


def _upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Rectangular pulse shaping (simple on purpose; RRC comes later)."""
    return np.repeat(symbols, sps)


BPSK = np.array([-1, 1], dtype=complex)
QPSK = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
_lv = np.array([-3, -1, 1, 3])
QAM16 = (_lv[:, None] + 1j * _lv[None, :]).ravel()


def gen_digital(n: int, constellation: np.ndarray, sps: int = 8,
                seed: int | None = None) -> np.ndarray:
    n_sym = int(np.ceil(n / sps))
    x = _upsample(_symbols(n_sym, constellation, seed), sps)[:n]
    return _normalize(x)


def gen_bpsk(n, sps=8, seed=None):
    return gen_digital(n, BPSK, sps, seed)


def gen_qpsk(n, sps=8, seed=None):
    return gen_digital(n, QPSK, sps, seed)


def gen_qam16(n, sps=8, seed=None):
    return gen_digital(n, QAM16, sps, seed)


GENERATORS = {
    "am": gen_am,
    "fm": gen_fm,
    "bpsk": gen_bpsk,
    "qpsk": gen_qpsk,
    "qam16": gen_qam16,
}


def add_awgn(x: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Add complex white Gaussian noise for the requested SNR (dB)."""
    sig_p = np.mean(np.abs(x) ** 2)
    noise_p = sig_p / (10 ** (snr_db / 10))
    rng = _rng(seed)
    noise = np.sqrt(noise_p / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return (x + noise).astype(np.complex64)


def generate(mod: str, n: int = 4096, snr_db: float = 20.0,
             seed: int | None = None) -> np.ndarray:
    if mod not in GENERATORS:
        raise ValueError(f"Unknown modulation '{mod}'. Choose from {MODULATIONS}.")
    clean = GENERATORS[mod](n, seed=seed)
    return add_awgn(clean, snr_db, seed=None if seed is None else seed + 1)


def save_iq(x: np.ndarray, path: str | Path) -> None:
    """Save as raw interleaved float32 I/Q (a.k.a. complex64 .iq file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x.astype(np.complex64).tofile(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic IQ file.")
    p.add_argument("--mod", choices=MODULATIONS, required=True)
    p.add_argument("--snr", type=float, default=20.0, help="SNR in dB")
    p.add_argument("--n", type=int, default=4096, help="number of samples")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    save_iq(generate(a.mod, a.n, a.snr, a.seed), a.out)
    print(f"Wrote {a.n} samples of {a.mod} @ {a.snr} dB SNR -> {a.out}")


if __name__ == "__main__":
    main()
