import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import time
from beamngpy import BeamNGpy, Scenario, Vehicle
from object_detector.object_detector import ObjectDetector
from object_detector.config_object_detector import DetectorConfig
from segmentation.models.resnet_unet import RESNET18_UNET
from lanedetector import LaneDetector

# BeamNG
BNG_HOME = r'C:\Simulators\BeamNG.drive'
BNG_USER = r"C:\Users\mike\AppData\Local\BeamNG.drive\0.32"
BNG_HOST = 'localhost'
BNG_PORT = 64256

# Scenario
SCENARIO_MAP = 'west_coast_usa'
SCENARIO_NAME = 'example'
VEHICLE_MODEL = 'etk800'
SPAWN_POS = (-819.472, -500.348, 106.633)
SPAWN_ROT = (0.002, 0.004, 0.923, -0.386)

# Camera (OBS)
CAMERA_ID = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CHECKPOINT_PATH = "segmentation/weights/residual_unet_weights.pth.tar"
MODEL_INPUT_HEIGHT = 240
MODEL_INPUT_WIDTH = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LANE_DETECTION
LANE_DETECTION_RATE_HZ = 5.0 
LANE_CONFIDENCE_THRESHOLD = 0.9

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
        """
        Runs the Neural Network inference.
        Returns the binary mask (0 or 255).
        """
        original_height, original_width = frame.shape[:2]
        
        # preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=frame_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        
        # inference
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
    #bng.scenario.start()
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

    # Initialize components
    seg_model = LaneSegmentationModel()
    
    # ROI: (x_start, y_start, width, height) as percentages of screen
    # Adjust 'roi_rect' to move the trigger box.
    detector = LaneDetector(
        roi_rect=(0.35, 0.65, 0.3, 0.30),
        history_length=3,
        trigger_confidence=0.66 
    )
    cnfg = DetectorConfig()
    obj_detector = ObjectDetector(cnfg)

    cap = init_camera()
    input("Press [Enter] to start lane detection...")
    bng.resume()
    
    prev_detection_time = 0
    detection_interval = 1.0 / LANE_DETECTION_RATE_HZ
    
    display_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    print(f"System running. Detection rate capped at {LANE_DETECTION_RATE_HZ}Hz. Press 'q' to exit.")

    while True:
        # We only update data from the scenario, we don't step physics (handled by game in real-time now)
        scenario.update()
        
        ret, frame = cap.read()
        if not ret: break


        current_time = time.time()
        obj_detector.detect_track_and_alert(frame)

        if current_time - prev_detection_time > detection_interval:
            mask = seg_model.predict(frame)
            
            detector.update(mask)
            
            prev_detection_time = current_time

        display_frame = detector.visualize_state(frame)
        
        if display_frame is not None:
            display_frame = cv2.resize(display_frame, (640, 360))
            cv2.imshow('Lane Departure Warning System', display_frame)

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