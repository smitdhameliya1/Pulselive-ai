import numpy as np
from pulselive.dsp import PulseEstimator, spectrum_peak, FS


def make_rgb(bpm, seconds=12, fs=30, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fs)
    pulse = np.sin(2 * np.pi * (bpm / 60) * t)
    base = np.array([150.0, 110.0, 95.0])
    rgb = np.tile(base, (len(t), 1))
    rgb[:, 1] += 0.5 * pulse
    rgb[:, 0] += 0.2 * pulse
    rgb += rng.normal(0, noise, rgb.shape)
    return t, rgb


def test_recovers_bpm():
    t, rgb = make_rgb(72)
    est = PulseEstimator()
    for ti, c in zip(t, rgb):
        est.add(ti, c)
    for _ in range(3):
        res = est.estimate()
    assert res["state"] == "LIVE"
    assert abs(res["bpm"] - 72) < 3


def test_pure_noise_is_not_live():
    rng = np.random.default_rng(1)
    t = np.arange(0, 12, 1 / 30)
    est = PulseEstimator(snr_threshold_db=3.0)
    for ti in t:
        est.add(ti, 120 + rng.normal(0, 1, 3))
    states = [est.estimate()["state"] for _ in range(4)]
    assert states[-1] != "LIVE"


def test_spectrum_peak_clean_sine():
    t = np.arange(0, 10, 1 / FS)
    bpm, snr = spectrum_peak(np.sin(2 * np.pi * 1.2 * t))
    assert abs(bpm - 72) < 1 and snr > 10


def test_provisional_bpm_appears_before_live_window():
    t, rgb = make_rgb(78, seconds=3.3, noise=0.15)
    est = PulseEstimator(window_sec=6, min_sec=3.5, provisional_sec=2.2,
                         required_good=2, snr_threshold_db=-2.0)
    for ti, c in zip(t, rgb):
        est.add(ti, c)
    res = est.estimate()
    assert res["state"] == "ANALYZING"
    assert res["provisional_bpm"] is not None
    assert abs(res["provisional_bpm"] - 78) < 6
