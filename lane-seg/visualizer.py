import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from vggunet import VGG_UNET

# Configuration
VIDEO_PATH = "samples/v1.mp4"
CHECKPOINT_PATH = "vgg_unet_bn.pth.tar"

class LaneSegmentationVisualizer:
    def __init__(self):
        # Load model
        self.model = VGG_UNET(in_channels=3, out_channels=1).to("cuda")
        checkpoint = torch.load(CHECKPOINT_PATH)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        
        # Define transforms
        self.transform = A.Compose([
            A.Resize(height=240, width=320),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2()
        ])
    
    def process_frame(self, frame):
        """Process a single frame and return the visualization"""
        original_height, original_width = frame.shape[:2]
        
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=frame_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to("cuda")
        
        # Get prediction
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img_tensor))
            pred = (pred > 0.9).float()
        
        # Convert to mask
        mask = pred.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (original_width, original_height))
        mask = (mask * 255).astype(np.uint8)
        
        # Create overlay
        overlay = frame.copy()
        overlay[mask > 127] = [0, 255, 0]  # Green for lanes
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result, mask
    
    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        cv2.namedWindow('Lane Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lane Segmentation', 1280, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result, mask = self.process_frame(frame)
            
            # Stack result and mask side by side
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            display = np.hstack([result, mask_colored])
            
            cv2.imshow('Lane Segmentation', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    visualizer = LaneSegmentationVisualizer()
    visualizer.run()