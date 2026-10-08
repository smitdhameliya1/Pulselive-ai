"""Signal processing: RGB traces -> heart rate (BPM) + signal quality (SNR).

Designed for responsive webcam rPPG.  A provisional BPM can appear after a
short warm-up, while LIVE is only asserted after a longer, quality-gated
window.  This is a wellness/demo estimate, not a medical measurement.
"""
from collections import deque
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend

LOW_HZ, HIGH_HZ = 0.7, 3.0          # 42-180 BPM
FS = 30.0                           # uniform resample rate


def bandpass(x: np.ndarray, fs: float = FS, low=LOW_HZ, high=HIGH_HZ) -> np.ndarray:
    sos = butter(3, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def chrom(rgb: np.ndarray, fs: float = FS) -> np.ndarray:
    """CHROM method (de Haan & Jeanne, 2013)."""
    norm = rgb / (rgb.mean(axis=0, keepdims=True) + 1e-9)
    r, g, b = norm[:, 0], norm[:, 1], norm[:, 2]
    xs = bandpass(3 * r - 2 * g, fs)
    ys = bandpass(1.5 * r + g - 1.5 * b, fs)
    alpha = np.std(xs) / (np.std(ys) + 1e-9)
    return xs - alpha * ys


def green(rgb: np.ndarray, fs: float = FS) -> np.ndarray:
    return bandpass(detrend(rgb[:, 1]), fs)


def _quadratic_peak(y: np.ndarray, i: int) -> float:
    """Sub-bin peak offset in [-0.5, 0.5] using a quadratic fit."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    den = a - 2.0 * b + c
    if abs(den) < 1e-18:
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def spectrum_peak(sig: np.ndarray, fs: float = FS, prior_bpm=None):
    """Return (bpm, snr_db) using a harmonic-aware spectral peak.

    Zero-padding + quadratic interpolation gives a smoother peak estimate; it
    does not create information that is absent from the time window.  A weak
    temporal prior reduces frame-to-frame peak hopping without preventing
    legitimate rate changes.
    """
    x = detrend(np.asarray(sig, dtype=float))
    x = (x - x.mean()) * np.hanning(len(x))

    # Enough zero padding for smooth interpolation without excessive work.
    nfft = max(2048, 1 << int(np.ceil(np.log2(max(8 * len(x), 2)))))
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    power = np.abs(np.fft.rfft(x, n=nfft)) ** 2
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    f, p = freqs[band], power[band]

    # Prefer a fundamental that also has energy at its second harmonic.
    score = p.copy()
    for j, fj in enumerate(f):
        h = 2.0 * fj
        if h <= HIGH_HZ:
            score[j] += 0.35 * np.interp(h, f, p)

    # Soft continuity prior: only a small preference, never a hard clamp.
    if prior_bpm is not None:
        delta = np.abs(f * 60.0 - float(prior_bpm))
        score *= 0.80 + 0.20 * np.exp(-0.5 * (delta / 15.0) ** 2)

    i = int(np.argmax(score))
    offset = _quadratic_peak(score, i)
    df = f[1] - f[0] if len(f) > 1 else 0.0
    peak_f = float(f[i] + offset * df)
    peak_f = float(np.clip(peak_f, LOW_HZ, HIGH_HZ))

    # SNR around the chosen fundamental plus its 2nd harmonic when present.
    sig_mask = np.abs(f - peak_f) <= 0.12
    if 2.0 * peak_f <= HIGH_HZ:
        sig_mask |= np.abs(f - 2.0 * peak_f) <= 0.12
    sig_pow = float(p[sig_mask].sum())
    noise_pow = float(p[~sig_mask].sum())
    snr_db = 10.0 * np.log10((sig_pow + 1e-12) / (noise_pow + 1e-12))
    return float(peak_f * 60.0), float(snr_db)


class PulseEstimator:
    def __init__(self, window_sec=8.0, min_sec=4.5, provisional_sec=2.8,
                 method="chrom", snr_threshold_db=1.0, fs=FS,
                 required_good=2, ema_alpha=0.35):
        self.window_sec = float(window_sec)
        self.min_sec = float(min_sec)
        self.provisional_sec = float(provisional_sec)
        self.method = method
        self.thr = float(snr_threshold_db)
        self.fs = float(fs)
        self.required_good = int(required_good)
        self.ema_alpha = float(ema_alpha)
        self.t = deque()
        self.rgb = deque()
        self.recent_bpm = deque(maxlen=5)
        self.good_streak = 0
        self.smoothed_bpm = None

    def add(self, t: float, rgb_mean):
        self.t.append(float(t))
        self.rgb.append(rgb_mean)
        while self.t and self.t[-1] - self.t[0] > self.window_sec:
            self.t.popleft()
            self.rgb.popleft()

    def reset(self):
        self.t.clear()
        self.rgb.clear()
        self.recent_bpm.clear()
        self.good_streak = 0
        self.smoothed_bpm = None

    def _duration(self):
        return (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def _resample(self):
        t = np.array(self.t, dtype=float)
        rgb = np.array(self.rgb, dtype=float)
        grid = np.arange(t[0], t[-1], 1.0 / self.fs)
        return np.stack([np.interp(grid, t, rgb[:, i]) for i in range(3)], axis=1)

    def estimate(self) -> dict:
        elapsed = self._duration()
        if len(self.t) < 10 or elapsed < self.provisional_sec:
            return {"state": "ANALYZING", "bpm": None, "provisional_bpm": None,
                    "snr": None, "signal": None,
                    "progress": min(1.0, elapsed / max(self.min_sec, 1e-9))}

        rgb = self._resample()
        # filtfilt needs a modest number of samples; guard tiny/irregular inputs.
        if len(rgb) < 30:
            return {"state": "ANALYZING", "bpm": None, "provisional_bpm": None,
                    "snr": None, "signal": None,
                    "progress": min(1.0, elapsed / max(self.min_sec, 1e-9))}

        sig = chrom(rgb, self.fs) if self.method == "chrom" else green(rgb, self.fs)
        bpm, snr = spectrum_peak(sig, self.fs, prior_bpm=self.smoothed_bpm)

        quality_ok = snr >= self.thr
        provisional = None
        if quality_ok:
            self.good_streak += 1
            self.recent_bpm.append(bpm)
            robust = float(np.median(self.recent_bpm))
            if self.smoothed_bpm is None:
                self.smoothed_bpm = robust
            else:
                a = self.ema_alpha
                self.smoothed_bpm = a * robust + (1.0 - a) * self.smoothed_bpm
            provisional = float(self.smoothed_bpm)
        else:
            self.good_streak = 0

        # Early readout is explicitly provisional; LIVE waits for enough time
        # and consecutive quality-gated estimates.
        if elapsed < self.min_sec:
            return {"state": "ANALYZING", "bpm": None,
                    "provisional_bpm": provisional, "raw_bpm": bpm,
                    "snr": snr, "signal": sig,
                    "progress": min(1.0, elapsed / self.min_sec)}

        live = self.good_streak >= self.required_good and self.smoothed_bpm is not None
        return {"state": "LIVE" if live else "NO PULSE",
                "bpm": float(self.smoothed_bpm) if live else None,
                "provisional_bpm": provisional,
                "raw_bpm": bpm, "snr": snr, "signal": sig, "progress": 1.0}
