from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(frozen=True)
class DetectorConfig:
    
    # S Tier Vars (Can make/break the model, must me fine-tuned)

    # --- Warning timing (seconds) ---
    soft_ttc_s: float = 5.0
    """SOFT warning threshold (seconds to collision).
    If a tracked vehicle's estimated TTC <= soft_ttc_s (but > hard_ttc_s),
    we play a soft warning (driver has time to react)."""

    hard_ttc_s: float = 4.0
    """HARD warning threshold (seconds to collision).
    If a tracked vehicle's estimated TTC <= hard_ttc_s, we play a hard warning."""

    # --- Proximity "too close" heuristics (vision-only proxy) ---
    too_close_area_frac: float = 0.12
    """Box area / frame area threshold that triggers HARD warning for any object.
    Example: 0.12 means if a detected object covers >=12% of the screen, it's considered too close."""

    too_close_center_frac: float = 0.20
    """Size of the 'center danger zone' as a fraction of frame width/height.
    Example: 0.20 creates a center rectangle of 20% width and 20% height.
    Objects inside this zone are treated as more threatening."""

    center_area_relax_factor: float = 0.60
    """If an object is inside the center danger zone, we allow a smaller area to still count as too close.
    Effective threshold becomes too_close_area_frac * center_area_relax_factor when inside center zone."""

    path_center_x_frac: float = 0.15
    """'In path' gate for vehicle TTC warnings.
    Vehicle must be within +/- (path_center_x_frac * frame_w) of screen center to count as in our path.
    Smaller => fewer false alarms, larger => more sensitive."""

    # --- TTC smoothing / stability ---
    ttc_ema_alpha: float = 0.35
    """EMA smoothing for box-height scale and ds/dt used in TTC estimation.
    Higher => more responsive but noisier. Lower => smoother but slower."""

    min_track_age_for_ttc: int = 4
    """Minimum number of frames a track must survive before we trust TTC enough to warn.
    Prevents warnings from one-frame flicker/noise."""

    # B Tier (These need to be set once, but once set they are forgotten and never ever touced, like ever)

    # --- YOLO inference ---
    model_path: str = "yolo26l.pt"
    """Path to YOLO model weights."""

    conf: float = 0.25
    """YOLO confidence threshold (higher => fewer detections)."""

    iou: float = 0.45
    """YOLO NMS IoU threshold (higher => more boxes survive NMS)."""

    device: Optional[str] = None
    """YOLO device string (e.g. 'cpu', 'cuda:0'). None lets ultralytics decide."""

    car_class_names: Tuple[str, ...] = ("car", "truck", "bus")
    """YOLO class names treated as 'vehicles' for TTC-based warning logic."""

    # --- Tracker matching ---
    track_iou_match_thresh: float = 0.30
    """IoU threshold to match a detection to an existing track.
    Higher => fewer ID switches but more missed matches."""

    track_max_missed: int = 10
    """How many consecutive frames a track can be missed before being deleted."""

    # F Tier (Litteral trash, never change it, does absolutly nothing, you are wasting your time)

    # --- Audio behavior ---
    soft_beep_freq: int = 1400
    """Frequency (Hz) for soft warning beep."""

    soft_beep_ms: int = 90
    """Duration (ms) for soft warning beep."""

    hard_beep_freq: int = 2600
    """Frequency (Hz) for hard warning beep."""

    hard_beep_ms: int = 180
    """Duration (ms) for hard warning beep."""

    soft_cooldown_s: float = 1.0
    """Minimum time between soft warnings (seconds)."""

    hard_cooldown_s: float = 0.4
    """Minimum time between hard warnings (seconds)."""

    # --- Numerical stability / clamps ---
    dt_min_s: float = 1e-3
    """Minimum dt used to avoid divide-by-zero when time deltas are tiny."""

    box_height_min_px: float = 1.0
    """Minimum box height in pixels used as a scale proxy (prevents zero)."""

    dsdt_min_px_s: float = 1e-3
    """Minimum ds/dt (pixels/sec). If ds/dt <= this, TTC is treated as infinity (not approaching)."""

    eps_area: float = 1e-6
    """Small epsilon to avoid division by zero in area calculations."""