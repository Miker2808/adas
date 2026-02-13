import cv2
import numpy as np
import time
import platform

# Cross-platform audio support
if platform.system() == 'Windows':
    import winsound
else:
    import os

class LaneDetector:

    def __init__(
        self,
        stencil_top = 0.5,
        stencile_bottom = 0.1,
        min_contour_area=2000,
        angle_range=(20, 160),
        hough_threshold=50,
        hough_min_line_length=50,
        hough_max_line_gap=30,
        poly_degree=2,
        num_fit_points=300,
        ldw_threshold=0.15,  # Percentage of middle area occupied to trigger warning
        ldw_persistence_time=1.5,  # Time in seconds before triggering warning
        middle_zone_width=0.3,  # Width of middle zone as fraction of image width
    ):
        self.stencil_top = stencil_top
        self.stencile_bottom = stencile_bottom
        self.min_contour_area = min_contour_area
        self.angle_range = angle_range
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.poly_degree = poly_degree
        self.num_fit_points = num_fit_points

        self.ldw_threshold = ldw_threshold
        self.ldw_persistence_time = ldw_persistence_time
        self.middle_zone_width = middle_zone_width
        
        # State tracking
        self.departure_start_time = None
        self.last_warning_time = None
        self.warning_cooldown = 3.0  # Cooldown between warnings in seconds

    def _apply_roi_stencil(self, mask, area = ()):
        h, w = mask.shape[:2]
        stencil = np.zeros((h, w), dtype=np.uint8)
        top_y = int(h * self.stencil_top)
        bottom_y = int(h * (1 - self.stencile_bottom))
        polygon = np.array([[0, bottom_y], [w, bottom_y], [w, top_y], [0, top_y]], dtype=np.int32)
        cv2.fillPoly(stencil, [polygon], 255)
        return cv2.bitwise_and(mask, stencil)

    def _remove_small_regions(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        valid_labels = np.where(areas >= self.min_contour_area)[0] + 1
        keep = np.isin(labels, valid_labels)
        result = np.zeros_like(mask)
        result[keep] = 255
        return result

    def _pick_nearest_lane(self, lines, center_x, side):
        if len(lines) == 0:
            return None
        bottom_xs = np.where(
            lines[:, 0, 1] > lines[:, 0, 3],
            lines[:, 0, 0],
            lines[:, 0, 2],
        ).astype(np.float64)
        if side == "left":
            distances = center_x - bottom_xs
        else:
            distances = bottom_xs - center_x
        # only consider lines on the correct side
        valid = distances > 0
        if not np.any(valid):
            return None
        distances_valid = np.where(valid, distances, np.inf)
        best_dist = np.min(distances_valid)
        threshold = best_dist * 2.5
        cluster_mask = valid & (distances_valid <= threshold)
        return lines[cluster_mask]


    def _discard_outer_edge_pixels(self, mask, center_x):
        h, w = mask.shape[:2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        cx_components = centroids[1:, 0]
        left_components = np.where(cx_components < center_x)[0] + 1
        right_components = np.where(cx_components >= center_x)[0] + 1
        keep_labels = []
        if len(left_components) > 0:
            left_xs = cx_components[left_components - 1]
            keep_labels.append(left_components[np.argmax(left_xs)])
        if len(right_components) > 0:
            right_xs = cx_components[right_components - 1]
            keep_labels.append(right_components[np.argmin(right_xs)])
        result = np.zeros_like(mask)
        if keep_labels:
            keep = np.isin(labels, keep_labels)
            result[keep] = 255
        return result

    def _play_mobileye_beep(self):
        """
        Play Mobileye-style warning beep (short double beep pattern).
        Mobileye uses a distinctive "beep-beep" sound with specific timing.
        """
        system = platform.system()
        
        if system == 'Windows':
            # Mobileye-style beep: Two short beeps at ~1000Hz
            frequency = 1000  # Hz
            duration_ms = 150  # milliseconds per beep
            
            # First beep
            winsound.Beep(frequency, duration_ms)
            time.sleep(0.08)  # Short gap between beeps
            # Second beep
            winsound.Beep(frequency, duration_ms)
            
        elif system == 'Darwin':  # macOS
            # Use say command with a quick alert sound
            # Mobileye pattern: two quick beeps
            os.system('afplay /System/Library/Sounds/Tink.aiff &')
            time.sleep(0.15)
            os.system('afplay /System/Library/Sounds/Tink.aiff &')
            
        else:  # Linux
            # Use beep command or paplay if available
            try:
                # Try using beep command (may need installation)
                os.system('beep -f 1000 -l 150 &')
                time.sleep(0.08)
                os.system('beep -f 1000 -l 150 &')
            except:
                # Fallback to system bell
                print('\a', end='', flush=True)
                time.sleep(0.08)
                print('\a', end='', flush=True)

    def lane_departure_warning(self, mask):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        h, w = mask.shape[:2]
        
        # Define middle zone (centered on image)
        middle_zone_half_width = int(w * self.middle_zone_width / 2)
        center_x = w // 2
        left_bound = center_x - middle_zone_half_width
        right_bound = center_x + middle_zone_half_width
        
        # Extract middle zone
        middle_zone = mask[:, left_bound:right_bound]
        
        # Calculate occupation percentage
        middle_zone_pixels = middle_zone.shape[0] * middle_zone.shape[1]
        occupied_pixels = np.sum(middle_zone > 0)
        occupation_ratio = occupied_pixels / middle_zone_pixels if middle_zone_pixels > 0 else 0
        
        current_time = time.time()
        departure_detected = occupation_ratio > self.ldw_threshold
        warning_triggered = False
        time_in_departure = None
        
        if departure_detected:
            # Lane detected in middle zone - potential departure
            if self.departure_start_time is None:
                # Start tracking departure
                self.departure_start_time = current_time
            
            # Calculate how long we've been in departure state
            time_in_departure = current_time - self.departure_start_time
            
            # Check if we should trigger warning
            if time_in_departure >= self.ldw_persistence_time:
                # Check cooldown period
                if (self.last_warning_time is None or 
                    (current_time - self.last_warning_time) >= self.warning_cooldown):
                    
                    # Trigger Mobileye-style warning
                    self._play_mobileye_beep()
                    self.last_warning_time = current_time
                    warning_triggered = True
        else:
            # No departure detected - reset timer
            self.departure_start_time = None
            time_in_departure = None
        
        return {
            'departure_detected': departure_detected,
            'warning_triggered': warning_triggered,
            'middle_occupation': occupation_ratio,
            'time_in_departure': time_in_departure
        }

    def detect(self, mask):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        binary = ((mask > 0).astype(np.uint8)) * 255
        h, w = binary.shape[:2]
        center_x = w // 2

        roi = self._apply_roi_stencil(binary)
        clean = self._remove_small_regions(roi)
        #clean = self._discard_outer_edge_pixels(clean, center_x)

        return clean
        
    def display_lines(self, image, lines):
        """
        display 2 lane lines
        """

        COLOR_RED = (0,0,255)
        COLOR_BLUE = (255,0,0)
        if lines[0].size != 0:
            x1_1, y1_1, x2_1, y2_1 = lines[0]
            cv2.line(image, (x1_1, y1_1), (x2_1, y2_1), COLOR_RED, 3)

        if lines[1].size != 0:
            x1_2, y1_2, x2_2, y2_2 = lines[1]
            cv2.line(image, (x1_2, y1_2), (x2_2, y2_2), COLOR_BLUE, 3)
        return image

    def draw_lanes(self, image, detection_result, color_left=(255, 0, 0), color_right=(0, 0, 255), thickness=3):
        overlay = image.copy()
        if detection_result["left_lane"] is not None:
            pts = detection_result["left_lane"].reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], False, color_left, thickness)
        if detection_result["right_lane"] is not None:
            pts = detection_result["right_lane"].reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], False, color_right, thickness)
        return overlay