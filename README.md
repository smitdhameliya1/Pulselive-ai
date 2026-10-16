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

