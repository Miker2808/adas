import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import time
from beamngpy import BeamNGpy, Scenario, Vehicle

# Import the model structure (ensure this file exists in your path)
from segmentation.models.resnet_unet import RESNET18_UNET
from lanedetector import LaneDetector

# --- Configuration ---
# BeamNG
BNG_HOME = r'E:\Games\BeamNG.drive'
BNG_USER = r"C:\Users\miker\AppData\Local\BeamNG.drive\0.32"
BNG_HOST = 'localhost'
BNG_PORT = 64256

# Scenario
SCENARIO_MAP = 'west_coast_usa'
SCENARIO_NAME = 'example'
VEHICLE_MODEL = 'etk800'
SPAWN_POS = (-819.472, -500.348, 106.633)
SPAWN_ROT = (0.002, 0.004, 0.923, -0.386)

# Camera / Model
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CHECKPOINT_PATH = "segmentation/weights/residual_unet_weights.pth.tar"
MODEL_INPUT_HEIGHT = 240
MODEL_INPUT_WIDTH = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Performance
DETECTION_RATE_HZ = 5.0  # Run AI detection 5 times per second
CRUISE_THROTTLE = 0.05

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
        
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=frame_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img_tensor))
            # Thresholding 0.9 provides high confidence, reducing noise
            pred = (pred >= 0.9).float() 
        
        # Post-process to mask
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
    bng.scenario.start()
    
    # Disable default AI so we can control it (or just let it coast)
    vehicle.ai.set_mode('disabled')
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

    cap = init_camera()
    input("Press [Enter] to start lane detection...")

    # Frequency Control Variables
    prev_detection_time = 0
    detection_interval = 1.0 / DETECTION_RATE_HZ
    
    # Store the last known frame/result to render when not detecting
    display_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    print(f"System running. Detection rate capped at {DETECTION_RATE_HZ}Hz. Press 'q' to exit.")

    while True:
        # 1. Update Simulation
        scenario.update()
        
        # 2. Get Camera Frame (Runs as fast as possible)
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()

        # 3. AI Detection Logic (Runs only at DETECTION_RATE_HZ)
        if current_time - prev_detection_time > detection_interval:
            # Run heavy model
            mask = seg_model.predict(frame)
            
            # Run logic and update state
            detector.update(mask)
            
            prev_detection_time = current_time

        # 4. Visualization (Runs every frame using latest state)
        # We pass the CURRENT frame, but the detector uses the stored LAST mask/state
        display_frame = detector.visualize_state(frame)
        
        if display_frame is not None:
            cv2.imshow('Lane Departure Warning System', display_frame)

        # 5. Controls
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("Paused. Press 'p' to unpause.")
            while cv2.waitKey(1) != ord('p'): pass
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    bng.close()

if __name__ == "__main__":
    main()