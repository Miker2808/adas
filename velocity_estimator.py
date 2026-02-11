# type: ignore
import cv2
import numpy as np
import time

class VelocityEstimator:
    """
    Vision-based ego velocity estimator using dense optical flow on the road surface.
    Uses Farneback dense flow which runs efficiently on a downscaled ROI crop.
    """

    def __init__(self):
        self.prev_gray_roi = None
        self.prev_time = None

        # Speed smoothing
        self.speed_history = []
        self.history_size = 8

        # Calibration: tune this for your camera setup
        # Higher = lower reported speed
        self.pixels_per_meter = 15.0

        # ROI crop coordinates (relative to image size)
        # We only compute flow on a small strip of road directly ahead
        self.roi_y_start = 0.65  # top of crop
        self.roi_y_end = 0.85    # bottom of crop
        self.roi_x_start = 0.3
        self.roi_x_end = 0.7

        # Downscale the ROI for speed
        self.flow_width = 160
        self.flow_height = 80

        # Farneback params (tuned for speed)
        self.fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        self.last_flow = None

    def _extract_roi(self, gray):
        """Crop and downscale the road region."""
        h, w = gray.shape
        y1 = int(h * self.roi_y_start)
        y2 = int(h * self.roi_y_end)
        x1 = int(w * self.roi_x_start)
        x2 = int(w * self.roi_x_end)
        crop = gray[y1:y2, x1:x2]
        small = cv2.resize(crop, (self.flow_width, self.flow_height), interpolation=cv2.INTER_AREA)
        return small

    def update(self, frame):
        """
        Feed a new BGR frame, returns estimated speed in km/h.
        @return: (speed_kph, flow_for_visualization_or_None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = self._extract_roi(gray)
        now = time.time()

        if self.prev_gray_roi is None:
            self.prev_gray_roi = roi
            self.prev_time = now
            return 0.0, None

        dt = now - self.prev_time
        if dt <= 0:
            dt = 1.0 / 30.0

        # Dense optical flow on tiny ROI
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray_roi, roi, None, **self.fb_params) 
        self.last_flow = flow

        # Get median flow vector (robust to outliers)
        flow_x = flow[..., 0].flatten()
        flow_y = flow[..., 1].flatten()

        # Filter out near-zero flows (static regions)
        magnitudes = np.sqrt(flow_x ** 2 + flow_y ** 2)
        moving_mask = magnitudes > 0.5
        if np.sum(moving_mask) < 20:
            self.prev_gray_roi = roi
            self.prev_time = now
            return self._smoothed(), flow

        med_x = np.median(flow_x[moving_mask])
        med_y = np.median(flow_y[moving_mask])
        pixel_speed = np.sqrt(med_x ** 2 + med_y ** 2)

        # Scale factor: the ROI was downscaled, so flow pixels are smaller
        h, w = frame.shape[:2]
        roi_actual_width = int(w * (self.roi_x_end - self.roi_x_start))
        scale = roi_actual_width / self.flow_width

        real_pixel_speed = pixel_speed * scale
        mps = (real_pixel_speed / self.pixels_per_meter) / dt
        kph = mps * 3.6
        kph = np.clip(kph, 0, 300)

        self.speed_history.append(float(kph))
        if len(self.speed_history) > self.history_size:
            self.speed_history.pop(0)

        self.prev_gray_roi = roi
        self.prev_time = now

        return self._smoothed(), flow

    def _smoothed(self):
        if len(self.speed_history) == 0:
            return 0.0
        return float(np.median(self.speed_history))