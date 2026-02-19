import argparse
import time

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from lanedetector import LaneDetector
from object_detector.config_object_detector import DetectorConfig
from object_detector.object_detector import ObjectDetector
from segmentation.models.resnet_unet import RESNET18_UNET

CHECKPOINT_PATH = "segmentation/weights/residual_unet_weights.pth.tar"
MODEL_INPUT_HEIGHT = 240
MODEL_INPUT_WIDTH = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANE_DETECTION_RATE_HZ = 5.0
LANE_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_VIDEO_PATH = "segmentation/sample2.mp4"


class LaneSegmentationModel:
    def __init__(self):
        print(f"Loading Model on {DEVICE}...")
        self.model = RESNET18_UNET(in_channels=3, out_channels=1).to(DEVICE)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.transform = A.Compose(
            [
                A.Resize(height=MODEL_INPUT_HEIGHT, width=MODEL_INPUT_WIDTH),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    def predict(self, frame):
        """
        Runs the Neural Network inference.
        Returns the binary mask (0 or 255).
        """
        original_height, original_width = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=frame_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = torch.sigmoid(self.model(img_tensor))
            pred = (pred >= LANE_CONFIDENCE_THRESHOLD).float()

        mask = pred.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (original_width, original_height))
        mask = (mask * 255).astype(np.uint8)
        return mask


def init_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    return cap


def parse_args():
    parser = argparse.ArgumentParser(description="Run lane/object detection using a source video file.")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=DEFAULT_VIDEO_PATH,
        help=f"Path to input video file (default: {DEFAULT_VIDEO_PATH})",
    )
    return parser.parse_args()


def main(video_path):
    seg_model = LaneSegmentationModel()

    detector = LaneDetector(
        roi_rect=(0.35, 0.65, 0.3, 0.30),
        history_length=3,
        trigger_confidence=0.66,
    )
    cnfg = DetectorConfig()
    obj_detector = ObjectDetector(cnfg)

    try:
        cap = init_video_capture(video_path)
    except Exception as e:
        print(f"Failed to open video source: {e}")
        return

    input("Press [Enter] to start lane detection...")

    prev_detection_time = 0.0
    detection_interval = 1.0 / LANE_DETECTION_RATE_HZ

    print(f"System running on {video_path}. Detection rate capped at {LANE_DETECTION_RATE_HZ}Hz. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        obj_detector.detect_track_and_alert(frame)

        if current_time - prev_detection_time > detection_interval:
            mask = seg_model.predict(frame)
            detector.update(mask)
            prev_detection_time = current_time

        display_frame = detector.visualize_state(frame)
        if display_frame is not None:
            display_frame = cv2.resize(display_frame, (640, 360))
            cv2.imshow("Lane Departure Warning System", display_frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("p"):
            print("Paused. Press 'p' to unpause.")
            while cv2.waitKey(1) != ord("p"):
                pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args.video_path)
