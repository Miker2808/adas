import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from vggunet import VGG_UNET

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Configuration
CHECKPOINT_PATH = "model/vgg_unet_bn.pth.tar"
IMAGE_TOPIC = "/sensing/camera/front/image_raw"

class LaneSegmentationNode(Node):
    def __init__(self):
        super().__init__('lane_segmentation_node')
        
        # ROS2 setup
        self.subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10
        )
        
        # Load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VGG_UNET(in_channels=3, out_channels=1).to(self.device)
        checkpoint = torch.load(CHECKPOINT_PATH)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        
        # Define transforms
        self.transform = A.Compose([
            A.Resize(height=240, width=320),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2()
        ])
        
        # Create window
        cv2.namedWindow('Lane Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lane Segmentation', 1280, 480)
        
        self.get_logger().info(f'Lane segmentation node started. Subscribing to {IMAGE_TOPIC}')
    
    def ros_to_cv2(self, msg):
        """Convert ROS2 Image message to OpenCV format without cv_bridge"""
        if msg.encoding == 'rgb8':
            channels = 3
            dtype = np.uint8
        elif msg.encoding == 'bgr8':
            channels = 3
            dtype = np.uint8
        elif msg.encoding == 'mono8':
            channels = 1
            dtype = np.uint8
        elif msg.encoding == '32FC1':
            channels = 1
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")
        
        # Convert message data to numpy array
        img_array = np.frombuffer(msg.data, dtype=dtype)
        
        if channels == 1:
            img_array = img_array.reshape((msg.height, msg.width))
        else:
            img_array = img_array.reshape((msg.height, msg.width, channels))
        
        # Convert RGB to BGR if needed (OpenCV uses BGR)
        if msg.encoding == 'rgb8':
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    
    def process_frame(self, frame):
        """Process a single frame and return the visualization"""
        original_height, original_width = frame.shape[:2]
        
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=frame_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img_tensor))
            pred = (pred > 0.99).float()
        
        # Convert to mask
        mask = pred.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (original_width, original_height))
        mask = (mask * 255).astype(np.uint8)
        
        # Create overlay
        overlay = frame.copy()
        overlay[mask > 127] = [0, 255, 0]  # Green for lanes
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result, mask
    
    def image_callback(self, msg):
        """Callback for receiving images from ROS2 topic"""
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.ros_to_cv2(msg)
            
            # Process frame
            result, mask = self.process_frame(frame)
            
            # Stack result and mask side by side
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            display = np.hstack([result, mask_colored])
            
            # Display
            cv2.imshow('Lane Segmentation', display)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    node = LaneSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()