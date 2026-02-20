from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DetectorConfig:
    
    ttc_soft_warning_seconds: float = 5.0
    ttc_hard_warning_seconds: float = 3.0

    danger_zone_screen_coverage_ratio: float = 0.2
    center_danger_zone_size_ratio: float = 0.20
    center_zone_coverage_multiplier: float = 0.60
    ego_path_width_ratio: float = 0.15

    time_to_collision_smoothing_factor: float = 0.35
    minimum_frames_to_trust_track: int = 4

    yolo_model_path: str = "yolo26l.pt"
    yolo_confidence_threshold: float = 0.25
    yolo_nms_iou_threshold: float = 0.45
    yolo_compute_device: Optional[str] = None
    target_vehicle_classes: Tuple[str, ...] = ("car", "truck", "bus")

    tracker_iou_matching_threshold: float = 0.30
    tracker_max_missed_frames: int = 10

    enable_debug_visualization: bool = True
    debug_window_title: str = "Object Detector Debug"

    audio_soft_warning_frequency_hz: int = 1400
    audio_soft_warning_duration_ms: int = 90
    audio_hard_warning_frequency_hz: int = 2600
    audio_hard_warning_duration_ms: int = 180
    
    audio_soft_warning_cooldown_seconds: float = 1.0
    audio_hard_warning_cooldown_seconds: float = 0.4

    math_min_time_delta_seconds: float = 1e-3
    math_min_bounding_box_height_pixels: float = 1.0
    math_min_approach_speed_pixels_per_second: float = 3.0
    math_min_box_height_for_warning_pixels: float = 40.0
    math_epsilon_area: float = 1e-6

    minimum_ego_speed_for_warnings_m_s: float = 1.5