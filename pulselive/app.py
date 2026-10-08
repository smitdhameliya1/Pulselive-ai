"""Real-time webcam heart-rate + liveness demo.

Usage:
    python -m pulselive.app
    python -m pulselive.app --profile fast
    python -m pulselive.app --profile stable
    python -m pulselive.app --video clip.mp4
"""
import argparse, csv, time
from collections import deque
import cv2
import numpy as np
from pulselive.dsp import PulseEstimator
from pulselive.roi import FaceROI

COLORS = {"LIVE": (80, 200, 80), "NO PULSE": (60, 60, 230),
          "ANALYZING": (0, 200, 255), "NO FACE": (150, 150, 150)}

PROFILES = {
    # Fastest useful UX.  More jitter / more sensitivity to motion.
    "fast": dict(window_sec=6.0, min_sec=3.5, provisional_sec=2.2,
                 required_good=2, update_sec=0.20, ema_alpha=0.45),
    # Recommended compromise for a normal 30-fps laptop webcam.
    "balanced": dict(window_sec=8.0, min_sec=4.5, provisional_sec=2.8,
                     required_good=2, update_sec=0.30, ema_alpha=0.35),
    # Closest to the original conservative behavior.
    "stable": dict(window_sec=10.0, min_sec=6.0, provisional_sec=4.0,
                   required_good=3, update_sec=0.50, ema_alpha=0.25),
}


def draw_wave(frame, sig, color):
    h, w = frame.shape[:2]
    y0, hh = h - 90, 70
    cv2.rectangle(frame, (10, y0), (w - 10, y0 + hh), (30, 30, 30), -1)
    if sig is None or len(sig) < 2:
        return
    s = sig[-int(w * 0.5):] if len(sig) > w * 0.5 else sig
    s = (s - s.min()) / (np.ptp(s) + 1e-9)
    xs = np.linspace(14, w - 14, len(s)).astype(int)
    ys = (y0 + hh - 6 - s * (hh - 12)).astype(int)
    cv2.polylines(frame, [np.stack([xs, ys], 1)], False, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="video file instead of webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--method", choices=["chrom", "green"], default="chrom")
    ap.add_argument("--snr", type=float, default=1.0, help="quality/liveness SNR threshold (dB)")
    ap.add_argument("--profile", choices=list(PROFILES), default="balanced")
    ap.add_argument("--log", help="CSV path to log results")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")

    # A stable 30fps capture helps rPPG. These are best-effort; cameras may ignore them.
    if not a.video:
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cfg = PROFILES[a.profile].copy()
    update_sec = cfg.pop("update_sec")
    roi = FaceROI()
    est = PulseEstimator(method=a.method, snr_threshold_db=a.snr, **cfg)

    last_est, result, frame_i, t0 = 0.0, {"state": "NO FACE"}, 0, time.perf_counter()
    motion = deque(maxlen=15)
    prev_nose = None
    writer = None
    f = None
    if a.log:
        f = open(a.log, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["t", "state", "bpm", "provisional_bpm", "raw_bpm", "snr_db", "motion"])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_i / file_fps if a.video else time.perf_counter() - t0
        frame_i += 1
        if not a.video:
            frame = cv2.flip(frame, 1)

        out = roi.process(frame)
        if out is None:
            est.reset()
            result = {"state": "NO FACE"}
            prev_nose = None
            motion.clear()
        else:
            if prev_nose is not None:
                motion.append(float(np.linalg.norm(out["nose"] - prev_nose) / max(out["face_w"], 1e-6)))
            prev_nose = out["nose"]
            motion_level = float(np.mean(motion)) if motion else 0.0

            # Large head movement corrupts the tiny color signal.  Reset only on
            # strong motion; mild motion is merely warned about.
            if motion_level > 0.035:
                est.reset()
                result = {"state": "ANALYZING", "bpm": None, "snr": None,
                          "signal": None, "progress": 0.0}
            else:
                est.add(t, out["rgb"])
                if t - last_est >= update_sec:
                    last_est = t
                    result = est.estimate()
                    if writer:
                        writer.writerow([
                            round(t, 3), result["state"], result.get("bpm"),
                            result.get("provisional_bpm"), result.get("raw_bpm"),
                            result.get("snr"), round(motion_level, 5)
                        ])

            for x1, y1, x2, y2 in out["boxes"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

        state = result["state"]
        col = COLORS[state]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 78), (20, 20, 20), -1)
        cv2.putText(frame, state, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)

        if state == "LIVE" and result.get("bpm") is not None:
            cv2.putText(frame, f"{result['bpm']:.0f} BPM", (15, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        elif state == "ANALYZING":
            provisional = result.get("provisional_bpm")
            if provisional is not None:
                text = f"~{provisional:.0f} BPM provisional - hold still"
            else:
                text = f"calibrating {result.get('progress', 0)*100:.0f}% - hold still"
            cv2.putText(frame, text, (15, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 1)

        if result.get("snr") is not None:
            cv2.putText(frame, f"SNR {result['snr']:.1f} dB", (frame.shape[1] - 170, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
        if motion and np.mean(motion) > 0.010:
            cv2.putText(frame, "too much motion", (frame.shape[1] - 190, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 140, 255), 2)

        draw_wave(frame, result.get("signal"), col)
        cv2.putText(frame, "Camera rPPG wellness estimate - not a medical device",
                    (15, frame.shape[0] - 98), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (200, 200, 200), 1)
        cv2.imshow("PulseLive AI", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            est.reset()
            motion.clear()

    if f:
        f.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
