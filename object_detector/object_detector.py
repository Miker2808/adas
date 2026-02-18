import time
import math
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from object_detector.config_object_detector import DetectorConfig

try:
    import winsound
    _HAS_WINSOUND = True
except Exception:
    _HAS_WINSOUND = False

# -----------------------------
# Utilities
# -----------------------------
def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return np.array([x1, y1, w, h], dtype=np.float32)

def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)

def center_of_xyxy(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


#Track state
@dataclass
class Track:
    track_id: int
    cls_id: int
    last_xyxy: np.ndarray
    last_t: float
    age: int = 1
    missed: int = 0
    #track a proxy "scale" s (box height) and its derivative
    s_ema: float = 0.0
    s_prev: float = 0.0
    dsdt_ema: float = 0.0
    ttc_s: float = math.inf

    def update(self, i_xyxy: np.ndarray, t: float, i_config: DetectorConfig):
        dt = max(i_config.dt_min_s, t - self.last_t)
        x1, y1, x2, y2 = i_xyxy
        s = float(max(i_config.box_height_min_px, y2 - y1))  #box height proxy
        ema_alpha = i_config.ttc_ema_alpha

        if self.age == 1:
            self.s_ema = s
            self.s_prev = s
            self.dsdt_ema = 0.0
            self.ttc_s = math.inf
        else:
            s_ema_new = ema_alpha * s + (1.0 - ema_alpha) * self.s_ema
            dsdt = (s_ema_new - self.s_ema) / dt  #pixels per second
            self.dsdt_ema = ema_alpha * dsdt + (1.0 - ema_alpha) * self.dsdt_ema
            self.s_prev = self.s_ema
            self.s_ema = s_ema_new

            if self.dsdt_ema > i_config.dsdt_min_px_s:
                self.ttc_s = self.s_ema / self.dsdt_ema
            else:
                self.ttc_s = math.inf

        self.last_xyxy = i_xyxy
        self.last_t = t
        self.age += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1


class SimpleIoUTracker:
    """
    Minimal multi-object tracker:
    - Assign detections to existing tracks using IoU greedy matching.
    - Create new tracks for unmatched detections.
    - Drop tracks after max_missed frames.
    """
    def __init__(self, i_config: DetectorConfig):
        self.cfg = i_config
        self.iou_match_thresh = i_config.track_iou_match_thresh
        self.max_missed = i_config.track_max_missed
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, dets: List[Tuple[int, np.ndarray]], t: float) -> Dict[int, Track]:
        """
        dets: list of (cls_id, xyxy)
        returns: dict of active tracks
        """
        track_ids = list(self.tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(dets)))
        matches = []

        #Build IoU matrix
        if track_ids and dets:
            iou_mat = np.zeros((len(track_ids), len(dets)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                for j, (_, dxyxy) in enumerate(dets):
                    iou_mat[i, j] = box_iou_xyxy(self.tracks[tid].last_xyxy, dxyxy)

            #Greedy assignment
            while True:
                i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                best = float(iou_mat[i, j])
                if best < self.iou_match_thresh:
                    break
                tid = track_ids[i]
                matches.append((tid, j))
                iou_mat[i, :] = -1
                iou_mat[:, j] = -1
                if tid in unmatched_tracks:
                    unmatched_tracks.remove(tid)
                if j in unmatched_dets:
                    unmatched_dets.remove(j)

        #Update matched tracks
        for tid, j in matches:
            cls_id, xyxy = dets[j]
            tr = self.tracks[tid]
            tr.cls_id = cls_id
            tr.update(xyxy, t, self.cfg)

        #Mark unmatched tracks as missed, drop old ones
        to_delete = []
        for tid in list(unmatched_tracks):
            tr = self.tracks[tid]
            tr.mark_missed()
            if tr.missed > self.max_missed:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        #Create new tracks for unmatched detections
        for j in list(unmatched_dets):
            cls_id, xyxy = dets[j]
            tid = self._next_id
            self._next_id += 1
            tr = Track(track_id=tid, cls_id=cls_id, last_xyxy=xyxy, last_t=t)
            tr.update(xyxy, t, self.cfg)  #initializes
            self.tracks[tid] = tr

        return self.tracks


#Detector and Alert Logic
class ObjectDetector:
    """
    YOLO detector + IoU tracker + TTC-based alerting.
    """
    def __init__(self, i_config:DetectorConfig):
        self.cfg = i_config
        self.model = YOLO(self.cfg.model_path)
        self.conf = self.cfg.conf
        self.iou = self.cfg.iou
        self.device = self.cfg.device

        self.soft_ttc_s = self.cfg.soft_ttc_s
        self.hard_ttc_s = self.cfg.hard_ttc_s

        self._last_soft = 0.0
        self._last_hard = 0.0

        self.tracker = SimpleIoUTracker(self.cfg)

        # Map car-like class names to ids (from model)
        self.names = self.model.names  # dict: id->name in ultralytics
        self.vehicle_ids = {int(cid) for cid, name in self.names.items() if name in self.cfg.car_class_names}

    def detect(self, frame):
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        r = results[0]
        return [] if r.boxes is None else r.boxes

    def _boxes_to_dets(self, boxes) -> List[Tuple[int, np.ndarray, float]]:
        """
        returns list of (cls_id, xyxy, conf)
        """
        dets = []
        for b in boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = b.xyxy[0].cpu().numpy().astype(np.float32)
            dets.append((cls_id, xyxy, conf))
        return dets

    def _is_too_close(self, xyxy: np.ndarray, frame_w: int, frame_h: int) -> bool:
        x1, y1, x2, y2 = xyxy
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        frame_area = float(frame_w * frame_h) + self.cfg.eps_area
        area_frac = area / frame_area

        cx, cy = center_of_xyxy(xyxy)
        # "center danger zone" (a proxy for "in our lane / in front")
        center_w = frame_w * self.cfg.too_close_center_frac
        center_h = frame_h * self.cfg.too_close_center_frac
        in_center = (abs(cx - frame_w / 2) <= center_w / 2) and (abs(cy - frame_h / 2) <= center_h / 2)

        base = self.cfg.too_close_area_frac
        relax = self.cfg.center_area_relax_factor
        return (area_frac >= base) or (in_center and area_frac >= base * relax)

    def detect_track_and_alert(self, frame):
        """
        Returns:
          - raw boxes
          - tracks dict
          - alert_level: "none" | "soft" | "hard"
        """
        t = time.time()
        boxes = self.detect(frame)
        dets_raw = self._boxes_to_dets(boxes)

        frame_h, frame_w = frame.shape[:2]

        # For tracking, we ignore very low conf if you want (optional)
        dets_for_track = [(cls_id, xyxy) for (cls_id, xyxy, conf) in dets_raw]

        tracks = self.tracker.update(dets_for_track, t)

        # Decide alerts
        hard = False
        soft = False

        # 1) any object too close => HARD
        for cls_id, xyxy, conf in dets_raw:
            if self._is_too_close(xyxy, frame_w, frame_h):
                hard = True
                break

        # 2) vehicle TTC logic (only if we have enough history)
        # If any vehicle has TTC <= hard => HARD
        # Else if any has TTC in (hard, soft] => SOFT
        if not hard:
            for tr in tracks.values():
                if tr.cls_id in self.vehicle_ids and tr.age >= self.cfg.min_track_age_for_ttc:
                    cx, cy = center_of_xyxy(tr.last_xyxy)
                    in_path = abs(cx - frame_w / 2) < frame_w * self.cfg.path_center_x_frac

                    # HARD: imminent collision with a vehicle
                    if in_path and tr.ttc_s <= self.hard_ttc_s:
                        hard = True
                        break

                    # SOFT: soon-ish collision with a vehicle (but driver has time)
                    elif in_path and tr.ttc_s <= self.soft_ttc_s:

                        if in_path and not self._is_too_close(tr.last_xyxy, frame_w, frame_h):
                            soft = True

        # Rate-limited beeps
        now = t
        alert_level = "none"
        if hard and (now - self._last_hard) >= self.cfg.hard_cooldown_s:
            self._beep(self.cfg.hard_beep_freq, self.cfg.hard_beep_ms)
            self._last_hard = now
            alert_level = "hard"
        elif soft and (now - self._last_soft) >= self.cfg.soft_cooldown_s:
            self._beep(self.cfg.soft_beep_freq, self.cfg.soft_beep_ms)
            self._last_soft = now
            alert_level = "soft"

        return boxes, tracks, alert_level

    def _beep(self, freq, ms):
        if _HAS_WINSOUND:
            winsound.Beep(int(freq), int(ms))
        else:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", end="", flush=True)

