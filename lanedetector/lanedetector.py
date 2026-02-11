import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import math
from lanedetector.models.resnet_unet import RESNET18_UNET


class LaneDetector:
    def __init__(self, checkpoint_path="weights/residual_unet_weights.pth.tar"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RESNET18_UNET(in_channels=3, out_channels=1).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(height=480, width=640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # Pre-allocate tensor for inference (avoids repeated GPU allocation)
        self._input_tensor = None

    @torch.no_grad()
    def get_mask(self, image):
        """
        Run AI model and return float mask [0..1] at original resolution.
        Optimized: reuses GPU tensor, runs with torch.no_grad via decorator.
        """
        h, w = image.shape[:2]
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=frame_rgb)
        tensor = augmented["image"].unsqueeze(0)

        # Reuse GPU tensor to avoid allocation overhead
        if self._input_tensor is None or self._input_tensor.shape != tensor.shape:
            self._input_tensor = tensor.to(self.device)
        else:
            self._input_tensor.copy_(tensor)

        pred = torch.sigmoid(self.model(self._input_tensor))
        mask = pred.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return mask

    def detect(self, image):
        """
        Run lane segmentation. Returns:
        - mask_float: float mask [0..1] at original resolution
        - mask_binary: uint8 binary mask (0 or 255)
        """
        mask_float = self.get_mask(image)
        mask_binary = (mask_float > 0.99).astype(np.uint8) * 255
        return mask_float, mask_binary


class LaneDepartureDetector:
    """
    Uses the segmentation mask directly to detect lane departure.
    
    Instead of fitting lines through edges:
    1. Sample vertical columns across the bottom of the image
    2. Find where the lane mask ENDS on each side (left edge and right edge of the lane)
    3. Track how those edges move relative to vehicle center
    4. If the edge crosses into the center zone -> departure
    
    This naturally ignores walls/barriers because the AI was trained
    on lane markings, not walls. And we analyze the mask shape directly
    instead of converting to edges -> lines which loses the semantic meaning.
    """

    STATE_OK = 0
    STATE_WARNING = 1
    STATE_DEPARTURE = 2

    def __init__(self):
        # How many horizontal scan rows to average over
        self.scan_rows = 10
        # Where to scan (relative to image height)
        self.scan_y_start = 0.7  # start scanning here
        self.scan_y_end = 0.85   # end scanning here

        # Thresholds (fraction of image width from center)
        # If lane edge is within this fraction of center -> warning
        self.warning_zone = 0.12
        # If lane edge crosses this -> departure
        self.departure_zone = 0.05

        self.left_state = self.STATE_OK
        self.right_state = self.STATE_OK

        # Smoothing
        self.left_edge_history = []
        self.right_edge_history = []
        self.history_size = 5

        # Hold departure state
        self.left_depart_time = 0
        self.right_depart_time = 0
        self.hold_time = 0.8

    def _find_lane_edges(self, mask_binary):
        """
        Scan the mask horizontally across multiple rows to find
        the left and right edges of the lane surface.
        
        Returns (left_edge_x, right_edge_x) as fractions of image width [0..1].
        Returns (None, None) if no lane found.
        """
        h, w = mask_binary.shape

        y_start = int(h * self.scan_y_start)
        y_end = int(h * self.scan_y_end)

        if y_end <= y_start:
            return None, None

        # Take a horizontal band and average vertically
        band = mask_binary[y_start:y_end, :]
        column_density = np.mean(band, axis=0) / 255.0  # [0..1] per column

        # Threshold: a column is "lane" if > 30% of its pixels are lane
        lane_cols = column_density > 0.3

        if np.sum(lane_cols) < 10:
            return None, None

        # Find leftmost and rightmost lane columns
        lane_indices = np.where(lane_cols)[0]
        left_edge = lane_indices[0] / w
        right_edge = lane_indices[-1] / w

        return left_edge, right_edge

    def update(self, mask_binary):
        """
        Update departure state from a binary lane mask.
        @param mask_binary: uint8 mask (0 or 255) from LaneDetector
        @return: (left_state, right_state, left_edge, right_edge)
        """
        import time
        now = time.time()

        left_edge, right_edge = self._find_lane_edges(mask_binary)

        # Smooth edges
        if left_edge is not None:
            self.left_edge_history.append(left_edge)
        if len(self.left_edge_history) > self.history_size:
            self.left_edge_history.pop(0)

        if right_edge is not None:
            self.right_edge_history.append(right_edge)
        if len(self.right_edge_history) > self.history_size:
            self.right_edge_history.pop(0)

        smooth_left = np.median(self.left_edge_history) if len(self.left_edge_history) > 0 else None
        smooth_right = np.median(self.right_edge_history) if len(self.right_edge_history) > 0 else None

        # Center of image = 0.5
        # Left edge of lane close to 0.5 means lane is narrow on the left = drifting left
        # Right edge of lane close to 0.5 means lane is narrow on the right = drifting right

        # Check left: how far is the left lane edge from center?
        if smooth_left is not None:
            left_margin = 0.5 - smooth_left  # positive = left edge is left of center (good)
            if left_margin < self.departure_zone:
                self.left_state = self.STATE_DEPARTURE
                self.left_depart_time = now
            elif left_margin < self.warning_zone:
                self.left_state = self.STATE_WARNING
            else:
                if self.left_state == self.STATE_DEPARTURE and \
                   (now - self.left_depart_time) < self.hold_time:
                    pass
                else:
                    self.left_state = self.STATE_OK
        else:
            # No lane detected at all — could be off-road
            if self.left_state != self.STATE_DEPARTURE:
                self.left_state = self.STATE_OK

        # Check right: how far is the right lane edge from center?
        if smooth_right is not None:
            right_margin = smooth_right - 0.5  # positive = right edge is right of center (good)
            if right_margin < self.departure_zone:
                self.right_state = self.STATE_DEPARTURE
                self.right_depart_time = now
            elif right_margin < self.warning_zone:
                self.right_state = self.STATE_WARNING
            else:
                if self.right_state == self.STATE_DEPARTURE and \
                   (now - self.right_depart_time) < self.hold_time:
                    pass
                else:
                    self.right_state = self.STATE_OK
        else:
            if self.right_state != self.STATE_DEPARTURE:
                self.right_state = self.STATE_OK

        return self.left_state, self.right_state, smooth_left, smooth_right