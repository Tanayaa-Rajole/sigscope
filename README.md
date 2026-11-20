# SigScope 📡

A Python toolkit to load raw IQ / WAV signals, analyze their spectrum, and
classify the modulation scheme (AM, FM, BPSK, QPSK, 16-QAM).

> Status: 🚧 in progress. Built one day at a time, see the roadmap below.

## Roadmap

- [x] **Day 1**: Project setup + synthetic signal generator
- [ ] **Day 2**: IQ / WAV file loader + tests
- [ ] **Day 3**: FFT, PSD (Welch), spectrogram plots
- [ ] **Day 4**: Feature extraction (bandwidth, SNR, cumulants)
- [ ] **Day 5**: Modulation classifier + accuracy vs SNR
- [ ] **Day 6**: Streamlit web UI
- [ ] **Day 7**: Polish, CI, screenshots, `v1.0`

## Quick start

```bash
git clone <your-repo-url> && cd sigscope
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# generate a noisy QPSK signal at 10 dB SNR
python -m sigscope.generate --mod qpsk --snr 10 --n 4096 --out data/qpsk.iq

# run tests
pytest
```

## Project layout

```
sigscope/
  generate.py      # synthetic AM/FM/BPSK/QPSK/16-QAM + AWGN
tests/
  test_generate.py
data/              # generated files (git-ignored)
```

## Signal model

All signals are complex baseband, normalized to unit average power.
Noise is complex AWGN scaled to the requested SNR:
`noise_power = signal_power / 10^(SNR_dB/10)`.

## License

MIT
