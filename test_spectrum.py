import numpy as np
import pytest

from sigscope.generate import generate
from sigscope.loader import Signal
from sigscope.spectrum import (compute_spectrogram, fft_spectrum,
                               peak_frequency, plot_overview, welch_psd)


def tone(f0=1000.0, fs=8000.0, n=8192):
    t = np.arange(n) / fs
    return Signal(np.exp(2j * np.pi * f0 * t).astype(np.complex64), fs=fs)


def test_fft_peak_at_tone_frequency():
    sig = tone(1000)
    assert peak_frequency(sig) == pytest.approx(1000, abs=sig.fs / len(sig) * 1.5)


def test_negative_frequency_tone():
    sig = tone(-1500)
    assert peak_frequency(sig) == pytest.approx(-1500, abs=10)


def test_fft_axis_is_centered_and_sorted():
    f, m = fft_spectrum(tone())
    assert np.all(np.diff(f) > 0)
    assert f[0] < 0 < f[-1]
    assert len(f) == len(m)


def test_welch_peak_and_length():
    f, p = welch_psd(tone(2000), nperseg=1024)
    assert len(f) == 1024
    assert f[np.argmax(p)] == pytest.approx(2000, abs=8000 / 1024 * 1.5)


def test_welch_power_integrates_to_signal_power():
    sig = Signal(generate("qpsk", 20000, 20, seed=1), fs=1000.0)
    f, p_db = welch_psd(sig, nperseg=1024)
    psd = 10 ** (p_db / 10)
    total = np.sum(psd) * (f[1] - f[0])
    assert total == pytest.approx(sig.power, rel=0.05)


def test_spectrogram_shape_and_chirp_trend():
    fs, n = 8000.0, 16000
    t = np.arange(n) / fs
    # linear chirp 500 Hz -> 3000 Hz
    phase = 2 * np.pi * (500 * t + 0.5 * (2500 / t[-1]) * t**2)
    sig = Signal(np.exp(1j * phase).astype(np.complex64), fs=fs)
    f, tt, s = compute_spectrogram(sig, nperseg=256)
    assert s.shape == (len(f), len(tt))
    peak_track = f[np.argmax(s, axis=0)]
    assert peak_track[-1] > peak_track[0] + 1500


def test_raw_array_input_uses_unit_fs():
    x = generate("bpsk", 2048, 20, seed=0)
    f, _ = fft_spectrum(x)
    assert abs(f).max() <= 0.5


def test_plot_overview_saves_file(tmp_path):
    out = tmp_path / "o.png"
    plot_overview(Signal(generate("qpsk", 4096, 15, seed=2)), title="t", save=str(out))
    assert out.exists() and out.stat().st_size > 5000
