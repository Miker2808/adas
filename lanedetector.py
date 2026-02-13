import cv2
import numpy as np
import time
import winsound

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
        ldw_threshold=0.15,
        ldw_time=1.5,
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
        self.ldw_time = ldw_time
        self.departure_start = None

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

    def _beep(self):
        winsound.Beep(1000, 150)
        time.sleep(0.08)
        winsound.Beep(1000, 150)

    def lane_departure_warning(self, mask):
        w = mask.shape[1]
        middle = mask[:, w//3:2*w//3]
        
        if np.sum(middle > 0) / middle.size > self.ldw_threshold:
            if self.departure_start is None:
                self.departure_start = time.time()
            elif time.time() - self.departure_start >= self.ldw_time:
                self._beep()
                self.departure_start = None
        else:
            self.departure_start = None