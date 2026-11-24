"""Spectral analysis: FFT, Welch PSD, spectrogram, and plotting helpers."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .loader import Signal


def _samples_fs(sig) -> tuple[np.ndarray, float]:
    """Accept a Signal or a (samples, fs) style raw array with fs=1."""
    if isinstance(sig, Signal):
        return sig.samples, sig.fs
    return np.asarray(sig), 1.0


def fft_spectrum(sig, n_fft: int | None = None, window: str = "hann"):
    """Windowed FFT magnitude in dB, centered on 0 Hz.

    Returns (freqs, mag_db). freqs are in Hz (or cycles/sample if fs=1).
    """
    x, fs = _samples_fs(sig)
    n = len(x) if n_fft is None else n_fft
    w = sps.get_window(window, min(len(x), n))
    seg = x[: len(w)] * w
    spec = np.fft.fftshift(np.fft.fft(seg, n))
    mag = np.abs(spec) / np.sum(w)            # amplitude-corrected
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    return freqs, 20 * np.log10(mag + 1e-12)


def welch_psd(sig, nperseg: int = 1024, noverlap: int | None = None,
              window: str = "hann"):
    """Welch power spectral density, two-sided, centered on 0 Hz.

    Returns (freqs, psd_db) with psd in dB per Hz (relative to unit power).
    """
    x, fs = _samples_fs(sig)
    nperseg = min(nperseg, len(x))
    f, pxx = sps.welch(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, return_onesided=False,
                       detrend=False, scaling="density")
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)
    return f, 10 * np.log10(pxx + 1e-18)


def compute_spectrogram(sig, nperseg: int = 256, noverlap: int | None = None,
                        window: str = "hann"):
    """Spectrogram in dB. Returns (freqs, times, sxx_db) with shape (freq, time)."""
    x, fs = _samples_fs(sig)
    nperseg = min(nperseg, len(x))
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, sxx = sps.spectrogram(x, fs=fs, window=window, nperseg=nperseg,
                                noverlap=noverlap, return_onesided=False,
                                detrend=False, scaling="density", mode="psd")
    f = np.fft.fftshift(f)
    sxx = np.fft.fftshift(sxx, axes=0)
    return f, t, 10 * np.log10(sxx + 1e-18)


def peak_frequency(sig, n_fft: int | None = None) -> float:
    """Frequency (Hz) of the strongest spectral component."""
    f, mag = fft_spectrum(sig, n_fft)
    return float(f[np.argmax(mag)])


# --------------------------------------------------------------------------
# Plotting (matplotlib imported lazily so the analysis functions stay light)
# --------------------------------------------------------------------------
def _freq_unit(fs: float):
    if fs >= 1e6:
        return 1e6, "MHz"
    if fs >= 1e3:
        return 1e3, "kHz"
    if fs == 1.0:
        return 1.0, "cycles/sample"
    return 1.0, "Hz"


def plot_overview(sig, title: str | None = None, nperseg: int = 1024,
                  save: str | None = None, show: bool = False):
    """3-panel figure: FFT, Welch PSD, spectrogram."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, fs = _samples_fs(sig)
    div, unit = _freq_unit(fs)

    f1, m1 = fft_spectrum(sig)
    f2, p2 = welch_psd(sig, nperseg=nperseg)
    f3, t3, s3 = compute_spectrogram(sig, nperseg=min(256, max(16, len(x) // 8)))

    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.plot(f1 / div, m1, lw=0.8)
    ax1.set(title="FFT magnitude", xlabel=f"Frequency ({unit})", ylabel="dB")
    ax1.grid(alpha=0.3)

    ax2.plot(f2 / div, p2, lw=0.8, color="tab:orange")
    ax2.set(title="Welch PSD", xlabel=f"Frequency ({unit})", ylabel="dB/Hz")
    ax2.grid(alpha=0.3)

    vmax = np.max(s3)
    im = ax3.pcolormesh(t3, f3 / div, s3, shading="auto", cmap="viridis",
                        vmin=vmax - 60, vmax=vmax)
    ax3.set(title="Spectrogram", xlabel="Time (s)" if fs != 1.0 else "Time (samples)",
            ylabel=f"Frequency ({unit})")
    fig.colorbar(im, ax=ax3, label="dB/Hz")

    if title:
        fig.suptitle(title, fontsize=13)
    if save:
        fig.savefig(save, dpi=130)
    if show:
        plt.show()
    return fig


def main() -> None:
    import argparse

    from .loader import load_signal

    p = argparse.ArgumentParser(description="Plot FFT / PSD / spectrogram of a signal file.")
    p.add_argument("file")
    p.add_argument("--fmt", default="complex64", help="raw IQ format (ignored for .wav)")
    p.add_argument("--fs", type=float, default=1.0, help="sample rate in Hz (raw IQ only)")
    p.add_argument("--out", default="overview.png")
    a = p.parse_args()
    sig = load_signal(a.file, fmt=a.fmt, fs=a.fs)
    plot_overview(sig, title=a.file, save=a.out)
    print(f"Saved {a.out}  (peak at {peak_frequency(sig):.4g} Hz)")


if __name__ == "__main__":
    main()
