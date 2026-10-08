# PulseLive AI - Webcam Heart Rate + Liveness Detection (rPPG)

Your skin color changes imperceptibly with every heartbeat as blood volume pulses.
This project extracts that signal from a normal webcam in real time, estimates
heart rate, and uses the presence of a pulse as a **liveness check** (a printed photo has no pulse).

## Pipeline
webcam frame -> MediaPipe Face Mesh -> skin ROIs (forehead + cheeks) -> mean RGB per frame
-> resample to 30 Hz -> CHROM pulse signal -> 0.7-3 Hz bandpass -> FFT peak = BPM
-> peak SNR decides LIVE / NO PULSE

## Setup (Python 3.10 or 3.11)
```bash
python -m venv venv
venv\Scripts\activate         # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m pulselive.app
```
Tips: face the light (front/window light), sit still for ~10 seconds, avoid colored lighting.
Keys: `q` quit, `r` reset.

Other modes:
```bash
python -m pulselive.app --video clip.mp4
python -m pulselive.app --method green --snr 2.0 --log logs/run.csv
pytest
```

## Validate it (do this - it is your interview gold)
1. Wear a smartwatch / pulse oximeter, run the app 1 minute, log with `--log`.
2. Compare average BPM to the device. Report mean absolute error.
3. Test: bright vs dim light, still vs moving, photo of a face on a phone screen vs real face.

## Known limitations (be honest in interviews)
- A *video replay* of a real face contains a real pulse, so this alone is not full anti-spoofing.
  Combine with blink / head-pose challenge-response for production.
- Dark skin tones, low light, motion and compression reduce SNR.
- Wellness estimate only, not a medical device.

## Ideas to extend
Add HRV (beat-to-beat intervals), respiration rate, a Streamlit/WebRTC UI, a small CNN/transformer
(e.g. PhysNet) trained on UBFC-rPPG for comparison with CHROM, and a spoof-detection benchmark.

## Responsive profiles
The app now separates an early **provisional BPM** from a quality-gated **LIVE BPM**.

```bash
python -m pulselive.app --profile balanced   # recommended
python -m pulselive.app --profile fast       # faster, more jitter/motion sensitive
python -m pulselive.app --profile stable     # slower, more conservative
```

Typical behavior on a clean 30-fps webcam signal:
- `fast`: provisional readout after ~2.2 s, LIVE after ~3.5-4 s
- `balanced`: provisional after ~2.8 s, LIVE after ~4.5-5 s
- `stable`: provisional after ~4 s, LIVE after ~6-7 s

A webcam cannot provide a trustworthy truly instantaneous heart rate: rPPG needs several
pulse cycles to separate the heart-rate frequency from lighting and motion noise.  Use a
contact PPG / pulse oximeter / ECG-grade device when clinically reliable or genuinely
beat-to-beat heart rate is required.
