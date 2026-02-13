import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle

from segmentation.models.resnet_unet import RESNET18_UNET
from lanedetector import LaneDetector

# BeamNG Settings
BNG_HOME = r'E:\Games\BeamNG.drive'
BNG_USER = r"C:\Users\miker\AppData\Local\BeamNG.drive\0.32"
BNG_HOST = 'localhost'
BNG_PORT = 64256

# Scenario Settings
SCENARIO_MAP = 'west_coast_usa'
SCENARIO_NAME = 'example'
VEHICLE_MODEL = 'etk800'
SPAWN_POS = (-819.472, -500.348, 106.633)
SPAWN_ROT = (0.002, 0.004, 0.923, -0.386)

# Camera Settings
CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Model Settings
CHECKPOINT_PATH = "segmentation/weights/residual_unet_weights.pth.tar"
MODEL_INPUT_HEIGHT = 240
MODEL_INPUT_WIDTH = 320
DEVICE = "cuda"

# Control Settings
CRUISE_THROTTLE = 0.05

class LaneSegmentationVisualizer:
    def __init__(self):
        # Load model
        self.model = RESNET18_UNET(in_channels=3, out_channels=1).to(DEVICE)
        checkpoint = torch.load(CHECKPOINT_PATH)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        
        # Define transforms
        # Normalization std/mean kept internal as requested
        self.transform = A.Compose([
            A.Resize(height=MODEL_INPUT_HEIGHT, width=MODEL_INPUT_WIDTH),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2()
        ])
    
    def process_frame(self, frame):
        """Process a single frame and return the visualization"""
        original_height, original_width = frame.shape[:2]
        
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=frame_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        
        # Get prediction
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img_tensor))
            pred = (pred >= 0.9).float() 
        
        # Convert to mask
        mask = pred.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (original_width, original_height))
        mask = (mask * 255).astype(np.uint8)
        
        # Create overlay
        overlay = frame.copy()
        overlay[mask > 127] = [0, 255, 0]  # Green for lanes
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result, mask

def init_camera():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    return cap


def start_simulation():
    # Instantiate BeamNGpy instance
    bng = BeamNGpy(BNG_HOST, BNG_PORT, home=BNG_HOME, user=BNG_USER)
    
    # Launch BeamNG.tech
    bng.open()
    
    # Create scenario
    scenario = Scenario(SCENARIO_MAP, SCENARIO_NAME)

    # Create vehicle
    vehicle = Vehicle('ego_vehicle', model=VEHICLE_MODEL)
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    
    # Place files defining our scenario for the simulator to read
    scenario.make(bng)
    bng.settings.set_deterministic(60)
    
    # Load and start our scenario
    bng.scenario.load(scenario)
    bng.scenario.start()
    
    # Make the vehicle's AI span the map
    vehicle.ai.set_mode('disabled')

    return bng, scenario, vehicle

def main():
    print("Loading, please wait")

    bng, scenario, vehicle = start_simulation()
    
    # Initialize the visualizer model
    visualizer = LaneSegmentationVisualizer()
    detector = LaneDetector()


    input("Press [Enter] to start lane detection when ready...")
    cap = init_camera()
    
    while (True):
        scenario.update()
        ret, image_rgb = cap.read()
        
        # Process frame using the visualizer model logic
        result, mask = visualizer.process_frame(image_rgb)
        detection = detector.detect(mask)
        ldw_result = detector.lane_departure_warning(mask)

        if ldw_result != None and ldw_result['warning_triggered']:
            print("Lane departure warning!")

        cv2.imshow('Lane Segmentation', detection)

        # Apply throttle
        #vehicle.control(steering=0, throttle=CRUISE_THROTTLE)

        # wait 1ms to check if keystroke 'q' called to close
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('p'):
            print("Press [Enter] to unpause")
            while cv2.waitKey(1) != ord('p'):
                #vehicle.control(steering=0, throttle=0, brake=0.5)
                pass

    bng.close()


if __name__ == "__main__":
    main()