import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import time
import math
from beamngpy import BeamNGpy, Scenario, Vehicle

from object_detector.object_detector import ObjectDetector
from object_detector.config_object_detector import DetectorConfig
from segmentation.models.resnet_unet import RESNET18_UNET
from lanedetector import LaneDetector
from audio_manager import AudioManager

BNG_HOME = r'E:\Games\BeamNG.drive'
BNG_USER = r"C:\Users\miker\AppData\Local\BeamNG.drive\0.32"
BNG_HOST = 'localhost'
BNG_PORT = 64256

SCENARIO_MAP = 'west_coast_usa'
SCENARIO_NAME = 'example'
VEHICLE_MODEL = 'etk800'
SPAWN_POS = (-836.7, -501.05, 106.62)
SPAWN_ROT = (0.002, 0.004, 0.923, -0.386)

CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CHECKPOINT_PATH = "segmentation/weights/residual_unet_weights.pth.tar"
MODEL_INPUT_HEIGHT = 240
MODEL_INPUT_WIDTH = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANE_DETECTION_RATE_HZ = 10.0 
LANE_CONFIDENCE_THRESHOLD = 0.9
LANE_DETECTION_ROI = (0.3, 0.8, 0.4, 0.20)

class LaneSegmentationModel:
    def __init__(self):
        print(f"Loading Model on {DEVICE}...")
        self.model = RESNET18_UNET(in_channels=3, out_channels=1).to(DEVICE)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(height=MODEL_INPUT_HEIGHT, width=MODEL_INPUT_WIDTH),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2()
        ])
    
    def predict(self, frame):
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

def init_camera():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    return cap

def start_simulation():
    print("Connecting to BeamNG...")
    bng = BeamNGpy(BNG_HOST, BNG_PORT, home=BNG_HOME, user=BNG_USER)
    bng.open()
    
    scenario = Scenario(SCENARIO_MAP, SCENARIO_NAME)
    vehicle = Vehicle('ego_vehicle', model=VEHICLE_MODEL)
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    
    scenario.make(bng)
    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    vehicle.ai.set_mode('disabled')

    print("Spawning traffic...")
    time.sleep(1.0)
    bng.traffic.spawn(max_amount=5)
    bng.scenario.start()
    bng.pause()

    return bng, scenario, vehicle

def main():
    try:
        bng, scenario, vehicle = start_simulation()
    except Exception as e:
        print(f"Failed to start simulation: {e}")
        return

    seg_model = LaneSegmentationModel()
    
    detector = LaneDetector(
        roi_rect=LANE_DETECTION_ROI,
        history_length=5,
        trigger_confidence=0.8 
    )
    
    cnfg = DetectorConfig()
    obj_detector = ObjectDetector(cnfg)
    audio_manager = AudioManager()

    cap = init_camera()
    input("Press [Enter] to start ADAS system...")
    bng.resume()
    
    prev_detection_time = 0
    detection_interval = 1.0 / LANE_DETECTION_RATE_HZ
    
    print(f"System running. Press 'q' to exit.")

    while True:
        scenario.update()
        vehicle.poll_sensors()
        
        vel = vehicle.state['vel']
        ego_speed_m_s = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        
        ret, frame = cap.read()
        if not ret: 
            break

        current_time = time.time()
        
        obj_alert_level, combined_frame = obj_detector.detect_track_and_alert(frame, ego_speed_m_s)
        
        if obj_alert_level == "hard":
            audio_manager.play_beep(
                cnfg.audio_hard_warning_frequency_hz, 
                cnfg.audio_hard_warning_duration_ms, 
                cnfg.audio_hard_warning_cooldown_seconds
            )
        elif obj_alert_level == "soft":
            audio_manager.play_beep(
                cnfg.audio_soft_warning_frequency_hz, 
                cnfg.audio_soft_warning_duration_ms, 
                cnfg.audio_soft_warning_cooldown_seconds
            )

        if current_time - prev_detection_time > detection_interval:
            mask = seg_model.predict(frame)
            
            is_lane_warning = detector.update(
                mask, 
                ego_speed_m_s, 
                cnfg.minimum_ego_speed_for_warnings_m_s, 
                audio_manager
            )
            
            prev_detection_time = current_time

        combined_frame = detector.visualize_state(combined_frame)
        
        cv2.putText(combined_frame, f"Ego Speed: {ego_speed_m_s * 3.6:.1f} km/h", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if combined_frame is not None:
            display_frame = cv2.resize(combined_frame, (1280, 720))
            cv2.imshow('ADAS Unified Window', display_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("Paused. Press 'p' to unpause.")
            while cv2.waitKey(1) != ord('p'): pass
            
    cap.release()
    cv2.destroyAllWindows()
    bng.close()

if __name__ == "__main__":
    main()