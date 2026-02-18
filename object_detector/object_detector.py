import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

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

    def _box_area_frac(self, xyxy: np.ndarray, frame_w: int, frame_h: int) -> float:
        x1, y1, x2, y2 = xyxy
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        return area / (float(frame_w * frame_h) + self.cfg.eps_area)

    def _render_debug_frame(
        self,
        frame: np.ndarray,
        dets_raw: List[Tuple[int, np.ndarray, float]],
        tracks: Dict[int, Track],
        alert_level: str,
        too_close_ids: Set[int],
    ) -> np.ndarray:
        dbg = frame.copy()
        frame_h, frame_w = dbg.shape[:2]

        # Center danger zone
        center_w = int(frame_w * self.cfg.too_close_center_frac)
        center_h = int(frame_h * self.cfg.too_close_center_frac)
        cx0, cy0 = frame_w // 2, frame_h // 2
        cx1 = max(0, cx0 - center_w // 2)
        cy1 = max(0, cy0 - center_h // 2)
        cx2 = min(frame_w - 1, cx0 + center_w // 2)
        cy2 = min(frame_h - 1, cy0 + center_h // 2)
        cv2.rectangle(dbg, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
        cv2.putText(dbg, "center danger zone", (cx1, max(15, cy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Path gate
        path_half = int(frame_w * self.cfg.path_center_x_frac)
        px1 = max(0, cx0 - path_half)
        px2 = min(frame_w - 1, cx0 + path_half)
        cv2.line(dbg, (px1, 0), (px1, frame_h - 1), (255, 255, 0), 1)
        cv2.line(dbg, (px2, 0), (px2, frame_h - 1), (255, 255, 0), 1)
        cv2.putText(dbg, "path gate", (px1 + 4, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Raw detections
        for det_i, (cls_id, xyxy, conf) in enumerate(dets_raw):
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            name = self.names.get(int(cls_id), str(cls_id))
            area_frac = self._box_area_frac(xyxy, frame_w, frame_h)
            is_close = det_i in too_close_ids
            color = (0, 0, 255) if is_close else (0, 220, 0)

            cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
            cdx, cdy = center_of_xyxy(xyxy)
            cv2.circle(dbg, (int(cdx), int(cdy)), 3, (255, 255, 255), -1)

            label = f"D{det_i} {name} conf={conf:.2f} area={area_frac:.3f}"
            cv2.putText(dbg, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Tracker / TTC overlay
        for tid, tr in tracks.items():
            x1, y1, x2, y2 = [int(v) for v in tr.last_xyxy]
            tr_cx, _ = center_of_xyxy(tr.last_xyxy)
            in_path = abs(tr_cx - frame_w / 2) < frame_w * self.cfg.path_center_x_frac
            is_vehicle = tr.cls_id in self.vehicle_ids
            ready = tr.age >= self.cfg.min_track_age_for_ttc

            if is_vehicle and ready and in_path and tr.ttc_s <= self.hard_ttc_s:
                tcolor = (0, 0, 255)
            elif is_vehicle and ready and in_path and tr.ttc_s <= self.soft_ttc_s:
                tcolor = (0, 165, 255)
            elif is_vehicle:
                tcolor = (255, 220, 0)
            else:
                tcolor = (160, 160, 160)

            cv2.rectangle(dbg, (x1, y1), (x2, y2), tcolor, 1)
            ttc_txt = "inf" if math.isinf(tr.ttc_s) else f"{tr.ttc_s:.2f}s"
            cname = self.names.get(int(tr.cls_id), str(tr.cls_id))
            tlabel = f"T{tid} {cname} age={tr.age} miss={tr.missed} ttc={ttc_txt} in_path={int(in_path)}"
            cv2.putText(dbg, tlabel, (x1, min(frame_h - 6, y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, tcolor, 1, cv2.LINE_AA)

        # Status bar
        status_color = (0, 255, 0) if alert_level == "none" else ((0, 165, 255) if alert_level == "soft" else (0, 0, 255))
        cv2.rectangle(dbg, (0, 0), (frame_w - 1, 46), (20, 20, 20), -1)
        cv2.putText(dbg, f"alert={alert_level.upper()} dets={len(dets_raw)} tracks={len(tracks)}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)
        cv2.putText(dbg, f"hard_ttc={self.hard_ttc_s:.2f}s soft_ttc={self.soft_ttc_s:.2f}s area_thr={self.cfg.too_close_area_frac:.3f}",
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

        return dbg

    def _is_too_close(self, xyxy: np.ndarray, frame_w: int, frame_h: int) -> bool:
        area_frac = self._box_area_frac(xyxy, frame_w, frame_h)

        cx, cy = center_of_xyxy(xyxy)
        center_w = frame_w * self.cfg.too_close_center_frac
        center_h = frame_h * self.cfg.too_close_center_frac
        in_center = (abs(cx - frame_w / 2) <= center_w / 2) and (abs(cy - frame_h / 2) <= center_h / 2)

        base = self.cfg.too_close_area_frac
        relax = self.cfg.center_area_relax_factor
        return (area_frac >= base) or (in_center and area_frac >= base * relax)


    def detect_track_and_alert(self, frame):
        t = time.time()
        boxes = self.detect(frame)
        dets_raw = self._boxes_to_dets(boxes)

        frame_h, frame_w = frame.shape[:2]
        dets_for_track = [(cls_id, xyxy) for (cls_id, xyxy, conf) in dets_raw]
        tracks = self.tracker.update(dets_for_track, t)

        hard = False
        soft = False

        too_close_ids: Set[int] = set()
        for det_i, (_, xyxy, _) in enumerate(dets_raw):
            if self._is_too_close(xyxy, frame_w, frame_h):
                too_close_ids.add(det_i)
        if too_close_ids:
            hard = True

        if not hard:
            for tr in tracks.values():
                if tr.cls_id in self.vehicle_ids and tr.age >= self.cfg.min_track_age_for_ttc:
                    cx, cy = center_of_xyxy(tr.last_xyxy)
                    in_path = abs(cx - frame_w / 2) < frame_w * self.cfg.path_center_x_frac

                    if in_path and tr.ttc_s <= self.hard_ttc_s:
                        hard = True
                        break
                    elif in_path and tr.ttc_s <= self.soft_ttc_s:
                        if not self._is_too_close(tr.last_xyxy, frame_w, frame_h):
                            soft = True

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

        if self.cfg.debug_visualization:
            debug_frame = self._render_debug_frame(
                frame=frame,
                dets_raw=dets_raw,
                tracks=tracks,
                alert_level=alert_level,
                too_close_ids=too_close_ids,
            )
            cv2.imshow(self.cfg.debug_window_name, debug_frame)
            cv2.waitKey(1)

        return boxes, tracks, alert_level


    def _beep(self, freq, ms):
        if _HAS_WINSOUND:
            winsound.Beep(int(freq), int(ms))
        else:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", end="", flush=True)

