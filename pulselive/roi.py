"""Face landmarks -> skin regions (forehead + both cheeks) -> mean RGB."""
import cv2
import numpy as np
import mediapipe as mp


class FaceROI:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=False, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def process(self, frame_bgr):
        """Return dict(rgb=(3,), boxes=[(x1,y1,x2,y2)...], nose=(x,y), face_w=float) or None."""
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        P = lambda i: np.array([lm[i].x * w, lm[i].y * h])
        face_w = np.linalg.norm(P(234) - P(454))

        # forehead: between top of head (10) and eyebrows (105/334)
        brow = (P(105) + P(334)) / 2
        top = P(10)
        fc = np.array([brow[0], (top[1] + brow[1]) / 2 + 0.01 * face_w])
        boxes = [self._box(fc, 0.32 * face_w, 0.09 * face_w, w, h)]
        # cheeks
        for idx in (205, 425):
            boxes.append(self._box(P(idx), 0.13 * face_w, 0.13 * face_w, w, h))

        means = []
        for x1, y1, x2, y2 in boxes:
            if x2 - x1 < 4 or y2 - y1 < 4:
                return None
            patch = frame_bgr[y1:y2, x1:x2]
            means.append(patch.reshape(-1, 3).mean(axis=0)[::-1])  # BGR -> RGB
        return {"rgb": np.mean(means, axis=0), "boxes": boxes,
                "nose": P(1), "face_w": face_w}

    @staticmethod
    def _box(c, bw, bh, w, h):
        x1, x2 = int(max(0, c[0] - bw / 2)), int(min(w, c[0] + bw / 2))
        y1, y2 = int(max(0, c[1] - bh / 2)), int(min(h, c[1] + bh / 2))
        return (x1, y1, x2, y2)
