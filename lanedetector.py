import cv2
import numpy as np
import time
import winsound
from collections import deque

class LaneDetector:
    def __init__(
        self,
        roi_rect=(0.35, 0.7, 0.3, 0.25), 
        ldw_threshold=0.1,      # Sensitivity: % of white pixels in ROI to count as a "hit"
        history_length=3,       # BUFFER SIZE: 3 frames @ 5Hz = 0.6s window
        trigger_confidence=0.6, # Threshold: need 60% of history to be "hits" to trigger
        mask_overlay_opacity=0.3,
        roi_overlay_opacity=0.4
    ):
        self.roi_rect = roi_rect 
        self.ldw_threshold = ldw_threshold
        
        # Smart Analysis / Smoothing Settings
        self.history = deque(maxlen=history_length)
        self.trigger_confidence = trigger_confidence
        
        # Visualization settings
        self.mask_opacity = mask_overlay_opacity
        self.roi_opacity = roi_overlay_opacity
        
        # State
        self.is_warning_active = False
        self.latest_clean_mask = None
        self.last_beep_time = 0

    def clean_mask(self, mask):
        """Cleans noise from the mask using morphological operations."""
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
        return cleaned

    def _get_raw_detection(self, mask):
        """Returns True if the current single frame shows a lane departure."""
        h, w = mask.shape[:2]
        rx, ry, rw, rh = self.roi_rect
        
        x1 = int(rx * w)
        y1 = int(ry * h)
        x2 = int((rx + rw) * w)
        y2 = int((ry + rh) * h)
        
        roi = mask[y1:y2, x1:x2]
        total_pixels = roi.size
        
        if total_pixels == 0: return False

        white_pixels = cv2.countNonZero(roi)
        density = white_pixels / total_pixels
        
        return density > self.ldw_threshold

    def update(self, mask):
        """
        Main update loop.
        Adds current result to history buffer and calculates smoothed state.
        """
        self.latest_clean_mask = self.clean_mask(mask)
        
        # 1. Get Instant Result
        is_departure_frame = self._get_raw_detection(self.latest_clean_mask)
        
        # 2. Add to History Buffer
        # Store 1.0 for True, 0.0 for False
        self.history.append(1.0 if is_departure_frame else 0.0)
        
        # 3. Calculate Average Confidence
        # If history is [0, 1, 1], average is 0.66 -> Trigger
        # If history is [1, 0, 0], average is 0.33 -> Safe
        if len(self.history) > 0:
            confidence = sum(self.history) / len(self.history)
        else:
            confidence = 0.0

        # 4. Hysteresis Trigger (Smart Switch)
        # If we are NOT warning, we need high confidence to START (prevents false positives)
        if not self.is_warning_active:
            if confidence >= self.trigger_confidence:
                self.is_warning_active = True
                self._try_beep()
        # If we ARE warning, we need low confidence to STOP (prevents flickering)
        else:
            if confidence < (self.trigger_confidence / 2): # Wait until it's clearly safe
                self.is_warning_active = False

        return self.is_warning_active

    def visualize_state(self, frame_rgb):
        """Draws the overlay."""
        if frame_rgb is None: return None
        
        output = frame_rgb.copy()
        h, w = output.shape[:2]
        
        # Draw Mask
        if self.latest_clean_mask is not None:
            color_mask = np.zeros_like(output)
            # Yellow if warning, Cyan if safe
            mask_color = [0, 0, 255] if self.is_warning_active else [255, 255, 0]
            color_mask[self.latest_clean_mask > 0] = mask_color
            output = cv2.addWeighted(output, 1.0, color_mask, self.mask_opacity, 0)

        # Draw ROI Box
        rx, ry, rw, rh = self.roi_rect
        x1, y1 = int(rx * w), int(ry * h)
        x2, y2 = int((rx + rw) * w), int((ry + rh) * h)
        
        color = (0, 0, 255) if self.is_warning_active else (0, 255, 0)
        text = "LANE DEPARTURE!" if self.is_warning_active else "SAFE"

        # Calculate confidence for display
        conf = sum(self.history)/len(self.history) if self.history else 0.0
        
        # Overlay Box
        overlay_box = output.copy()
        cv2.rectangle(overlay_box, (x1, y1), (x2, y2), color, -1)
        output = cv2.addWeighted(overlay_box, self.roi_opacity, output, 1 - self.roi_opacity, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Text Info
        cv2.putText(output, f"{text} (Conf: {conf:.2f})", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output

    def _try_beep(self):
        """Non-blocking beep with cooldown"""
        now = time.time()
        if now - self.last_beep_time > 1.0: # Max 1 beep per second
            winsound.Beep(2000, 300) # Frequency, Duration
            self.last_beep_time = now