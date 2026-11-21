import numpy as np
import pytest
from scipy.io import wavfile

from sigscope.generate import generate, save_iq
from sigscope.loader import load_iq, load_signal, load_wav, normalize


def test_complex64_roundtrip(tmp_path):
    x = generate("qpsk", 1024, 20, seed=0)
    f = tmp_path / "a.iq"
    save_iq(x, f)
    sig = load_iq(f, norm=False)
    assert np.allclose(sig.samples, x)
    assert sig.meta["format"] == "complex64"


def test_normalize_unit_power():
    x = (np.random.default_rng(0).standard_normal(1000) * 7).astype(np.complex64)
    assert np.mean(np.abs(normalize(x)) ** 2) == pytest.approx(1.0, rel=1e-4)


def test_remove_dc():
    x = np.ones(100, dtype=np.complex64) * (3 + 2j) + generate("bpsk", 100, 40, seed=1)
    y = normalize(x, remove_dc=True)
    assert abs(np.mean(y)) < 1e-4


def test_int16_format(tmp_path):
    iq = np.array([[16384, -16384], [0, 32767]], dtype=np.int16)  # I,Q pairs
    f = tmp_path / "b.iq"
    iq.tofile(f)
    sig = load_iq(f, fmt="int16", norm=False)
    assert sig.samples[0] == pytest.approx(0.5 - 0.5j, abs=1e-4)
    assert len(sig) == 2


def test_uint8_format(tmp_path):
    iq = np.array([255, 0, 128, 127], dtype=np.uint8)
    f = tmp_path / "c.iq"
    iq.tofile(f)
    sig = load_iq(f, fmt="uint8", norm=False)
    assert sig.samples[0].real == pytest.approx(1.0, abs=1e-2)
    assert sig.samples[0].imag == pytest.approx(-1.0, abs=1e-2)


def test_max_samples_and_odd_length(tmp_path):
    x = generate("am", 500, 30, seed=2)
    f = tmp_path / "d.iq"
    save_iq(x, f)
    assert len(load_iq(f, max_samples=100)) == 100
    # append a dangling float: should be ignored, not crash
    with open(f, "ab") as fh:
        fh.write(np.float32(1.0).tobytes())
    assert len(load_iq(f)) == 500


def test_bad_format_and_empty(tmp_path):
    f = tmp_path / "e.iq"
    f.write_bytes(b"")
    with pytest.raises(ValueError):
        load_iq(f)
    with pytest.raises(ValueError):
        load_iq(f, fmt="nope")


def test_wav_stereo_iq(tmp_path):
    fs = 48000
    t = np.arange(4800) / fs
    i, q = np.cos(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 1000 * t)
    data = (np.stack([i, q], axis=1) * 30000).astype(np.int16)
    f = tmp_path / "s.wav"
    wavfile.write(f, fs, data)
    sig = load_wav(f)
    assert sig.fs == fs
    assert sig.meta["layout"] == "stereo-iq"
    assert sig.power == pytest.approx(1.0, rel=1e-3)
    assert np.allclose(np.abs(sig.samples), 1.0, atol=0.01)  # constant envelope tone


def test_wav_mono_analytic(tmp_path):
    fs = 8000
    t = np.arange(8000) / fs
    data = (np.cos(2 * np.pi * 500 * t) * 20000).astype(np.int16)
    f = tmp_path / "m.wav"
    wavfile.write(f, fs, data)
    sig = load_wav(f)
    assert sig.meta["layout"] == "mono-analytic"
    # analytic signal of a cosine has a single positive-frequency peak
    spec = np.abs(np.fft.fft(sig.samples))
    freqs = np.fft.fftfreq(len(sig), 1 / fs)
    assert freqs[np.argmax(spec)] == pytest.approx(500, abs=2)
    assert sig.duration == pytest.approx(1.0)


def test_load_signal_dispatch(tmp_path):
    x = generate("fm", 256, 20, seed=3)
    f = tmp_path / "x.cfile"
    save_iq(x, f)
    assert len(load_signal(f, fs=2e6)) == 256
    assert load_signal(f, fs=2e6).fs == 2e6
    with pytest.raises(ValueError):
        load_signal(tmp_path / "x.mp3")
