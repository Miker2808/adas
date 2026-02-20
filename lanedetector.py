import cv2
import numpy as np
from collections import deque

class LaneDetector:
    def __init__(
        self,
        roi_rect=(0.35, 0.7, 0.3, 0.25), 
        ldw_threshold=0.1,
        history_length=3,
        trigger_confidence=0.6,
        mask_overlay_opacity=0.3,
        roi_overlay_opacity=0.4,
        warning_freq_hz=2000,
        warning_duration_ms=300,
        warning_cooldown_s=1.0
    ):
        self.roi_rect = roi_rect 
        self.ldw_threshold = ldw_threshold
        
        self.history = deque(maxlen=history_length)
        self.trigger_confidence = trigger_confidence
        
        self.mask_opacity = mask_overlay_opacity
        self.roi_opacity = roi_overlay_opacity
        
        self.warning_freq_hz = warning_freq_hz
        self.warning_duration_ms = warning_duration_ms
        self.warning_cooldown_s = warning_cooldown_s
        
        self.is_warning_active = False
        self.latest_clean_mask = None
        self.confidence = 0.0

    def clean_mask(self, mask):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
        return cleaned

    def _get_raw_detection(self, mask):
        h, w = mask.shape[:2]
        rx, ry, rw, rh = self.roi_rect
        
        x1 = int(rx * w)
        y1 = int(ry * h)
        x2 = int((rx + rw) * w)
        y2 = int((ry + rh) * h)
        
        roi = mask[y1:y2, x1:x2]
        total_pixels = roi.size
        
        if total_pixels == 0: 
            return False

        white_pixels = cv2.countNonZero(roi)
        density = white_pixels / total_pixels
        
        return density > self.ldw_threshold

    def update(self, mask, ego_speed_m_s, min_speed_m_s, audio_manager):
        self.latest_clean_mask = self.clean_mask(mask)
        is_departure_frame = self._get_raw_detection(self.latest_clean_mask)
        self.history.append(1.0 if is_departure_frame else 0.0)
        
        if len(self.history) > 0:
            self.confidence = sum(self.history) / len(self.history)
        else:
            self.confidence = 0.0

        if not self.is_warning_active:
            if self.confidence >= self.trigger_confidence:
                self.is_warning_active = True
        else:
            if self.confidence < (self.trigger_confidence / 2):
                self.is_warning_active = False

        if self.is_warning_active and ego_speed_m_s >= min_speed_m_s:
            audio_manager.play_lane_warning(
                self.warning_freq_hz,
                self.warning_duration_ms,
                self.warning_cooldown_s
            )

        return self.is_warning_active

    def visualize_state(self, frame_rgb):
        if frame_rgb is None: 
            return None
        
        output = frame_rgb.copy()
        h, w = output.shape[:2]
        
        if self.latest_clean_mask is not None:
            color_mask = np.zeros_like(output)
            mask_color = [0, 0, 255] if self.is_warning_active else [255, 255, 0]
            color_mask[self.latest_clean_mask > 0] = mask_color
            output = cv2.addWeighted(output, 1.0, color_mask, self.mask_opacity, 0)

        rx, ry, rw, rh = self.roi_rect
        x1, y1 = int(rx * w), int(ry * h)
        x2, y2 = int((rx + rw) * w), int((ry + rh) * h)
        
        color = (0, 0, 255) if self.is_warning_active else (0, 255, 0)
        text = "LANE DEPARTURE!" if self.is_warning_active else "SAFE"
        
        overlay_box = output.copy()
        cv2.rectangle(overlay_box, (x1, y1), (x2, y2), color, -1)
        output = cv2.addWeighted(overlay_box, self.roi_opacity, output, 1 - self.roi_opacity, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        cv2.putText(output, f"{text} (Conf: {self.confidence:.2f})", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output