import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter


class CricketBallTracker:
    def __init__(self, video_path, ball_color='auto'):
        self.video_path = video_path
        self.model = YOLO('yolov8x.pt')

        # ---------------- KALMAN ---------------- #
        self.kf = self._init_kalman_filter()
        self.tracking_initialized = False

        # ---------------- STATE ---------------- #
        self.ball_confirmed = False
        self.confirm_count = 0
        self.missed_frames = 0

        # ---------------- PARAMS ---------------- #
        self.min_confidence = 0.3
        self.ball_class_id = 32

        self.MIN_SPEED = -1.0            # px / frame
        self.CONFIRM_FRAMES = 3          # consecutive moving frames
        self.MAX_PREDICT_FRAMES = 3      # freeze after this

        # ---------------- TRAJECTORY ---------------- #
        self.trajectory = []
        self.MAX_TRAJ_LEN = 200

    # ---------------- KALMAN ---------------- #
    def _init_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.R *= 10
        kf.Q *= 0.1
        kf.P *= 1000
        return kf

    # ---------------- FIXED ROI ---------------- #
    def _fixed_roi(self, w, h):
        return (
            int(0.10 * w), int(0.15 * h),
            int(0.90 * w), int(0.45 * h)
        )

    # ---------------- YOLO ---------------- #
    def detect_ball_yolo(self, frame):
        results = self.model(frame, verbose=False)
        best, best_conf = None, 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == self.ball_class_id and conf > self.min_confidence:
                    if conf > best_conf:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        best = ((x1 + x2) / 2, (y1 + y2) / 2)
                        best_conf = conf
        return best

    # ---------------- COLOR ---------------- #
    def detect_ball_by_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100 or area > 330:
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri ** 2)
            if circ > 0.5:
                M = cv2.moments(c)
                score = circ * np.sqrt(area)
                if score > best_score:
                    best = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                    best_score = score
        return best

    # ---------------- MOTION CHECK ---------------- #
    def _is_moving(self):
        vx, vy = self.kf.x[2], self.kf.x[3]
        return np.hypot(vx, vy) >= self.MIN_SPEED

    # ---------------- MAIN ---------------- #
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {self.video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        video_stem = Path(self.video_path).stem

        out = cv2.VideoWriter(
            str(results_dir / f"{video_stem}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (w, h)
        )

        if not out.isOpened():
            out = cv2.VideoWriter(
                str(results_dir / f"{video_stem}.avi"),
                cv2.VideoWriter_fourcc(*'MJPG'),
                fps, (w, h)
            )

        annotations = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rx1, ry1, rx2, ry2 = self._fixed_roi(w, h)
            roi = frame[ry1:ry2, rx1:rx2]

            det = self.detect_ball_yolo(roi) or self.detect_ball_by_color(roi)

            if det:
                cx, cy = det
                cx += rx1
                cy += ry1

                if not self.tracking_initialized:
                    self.kf.x = np.array([cx, cy, 0, 0])
                    self.tracking_initialized = True
                else:
                    self.kf.predict()
                    self.kf.update(np.array([cx, cy]))

                if self._is_moving():
                    self.confirm_count += 1
                else:
                    self.confirm_count = 0

                if self.confirm_count >= self.CONFIRM_FRAMES:
                    self.ball_confirmed = True

                if self.ball_confirmed:
                    sx, sy = int(self.kf.x[0]), int(self.kf.x[1])
                    self.trajectory.append((sx, sy))
                    self.trajectory = self.trajectory[-self.MAX_TRAJ_LEN:]
                    annotations.append({'frame': frame_idx, 'x': sx, 'y': sy, 'visible': 1})
                    self.missed_frames = 0
                else:
                    annotations.append({'frame': frame_idx, 'x': -1, 'y': -1, 'visible': 0})
            else:
                self.missed_frames += 1
                if self.missed_frames > self.MAX_PREDICT_FRAMES:
                    self.ball_confirmed = False
                    self.confirm_count = 0
                    self.tracking_initialized = False
                    # self.trajectory.clear()
                annotations.append({'frame': frame_idx, 'x': -1, 'y': -1, 'visible': 0})

            # 🔵 past positions
            for pt in self.trajectory[:-1]:
                cv2.circle(frame, pt, 3, (255, 0, 0), -1)

            # 🟢 current
            if self.trajectory:
                cv2.circle(frame, self.trajectory[-1], 6, (0, 255, 0), -1)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        self._save_annotations(annotations)
        return annotations

    # ---------------- SAVE ---------------- #
    def _save_annotations(self, ann):
        out_dir = Path("annotations")
        out_dir.mkdir(exist_ok=True)
        name = Path(self.video_path).stem
        pd.DataFrame(ann).to_csv(out_dir / f"{name}.csv", index=False)
        print(f"Annotations saved at: annotations/{name}.csv")
