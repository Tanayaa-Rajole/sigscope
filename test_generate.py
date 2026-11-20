import numpy as np
import pytest

from sigscope.generate import MODULATIONS, add_awgn, generate, save_iq


@pytest.mark.parametrize("mod", MODULATIONS)
def test_shape_and_dtype(mod):
    x = generate(mod, n=1000, snr_db=20, seed=0)
    assert x.shape == (1000,)
    assert x.dtype == np.complex64


@pytest.mark.parametrize("mod", MODULATIONS)
def test_unit_power_at_high_snr(mod):
    x = generate(mod, n=8192, snr_db=60, seed=1)
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=0.05)


def test_snr_is_respected():
    clean = generate("qpsk", n=50000, snr_db=100, seed=2)
    noisy = add_awgn(clean, 10, seed=3)
    noise_p = np.mean(np.abs(noisy - clean) ** 2)
    measured = 10 * np.log10(np.mean(np.abs(clean) ** 2) / noise_p)
    assert measured == pytest.approx(10, abs=0.3)


def test_seed_reproducible():
    a = generate("bpsk", 512, 10, seed=5)
    b = generate("bpsk", 512, 10, seed=5)
    assert np.array_equal(a, b)


def test_unknown_mod():
    with pytest.raises(ValueError):
        generate("foo")


def test_save_iq_roundtrip(tmp_path):
    x = generate("qam16", 256, 15, seed=7)
    f = tmp_path / "x.iq"
    save_iq(x, f)
    y = np.fromfile(f, dtype=np.complex64)
    assert np.array_equal(x, y)
