import cv2
import numpy as np

class ADASHUD:
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        
        self.sky_color = (25, 25, 25)
        self.ground_color = (40, 40, 40)
        self.text_color = (255, 255, 255)
        
        self.safe_color = (0, 200, 0)
        self.soft_color = (0, 165, 255)
        self.hard_color = (0, 0, 255)
        
        self.horizon_y = int(self.height * 0.45)

    def _draw_background(self, hud):
        cv2.rectangle(hud, (0, 0), (self.width, self.horizon_y), self.sky_color, -1)
        cv2.rectangle(hud, (0, self.horizon_y), (self.width, self.height), self.ground_color, -1)

    def _draw_lanes(self, hud, is_lane_departure):
        lane_color = self.hard_color if is_lane_departure else self.safe_color
        
        left_top = (int(self.width * 0.4), self.horizon_y)
        left_bottom = (int(self.width * 0.1), self.height)
        
        right_top = (int(self.width * 0.6), self.horizon_y)
        right_bottom = (int(self.width * 0.9), self.height)
        
        cv2.line(hud, left_top, left_bottom, lane_color, 4)
        cv2.line(hud, right_top, right_bottom, lane_color, 4)

    def _draw_vehicle_and_alerts(self, hud, alert_level):
        center_x = self.width // 2
        
        if alert_level == "hard":
            car_w, car_h = 80, 60
            car_y = self.horizon_y + 10
            color = self.hard_color
            cv2.putText(hud, "BRAKE", (center_x - 55, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        elif alert_level == "soft":
            car_w, car_h = 50, 40
            car_y = self.horizon_y
            color = self.soft_color
            cv2.putText(hud, "VEHICLE AHEAD", (center_x - 85, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            car_w, car_h = 30, 20
            car_y = self.horizon_y - 10
            color = self.safe_color

        if alert_level in ["soft", "hard"]:
            pt1 = (center_x - car_w // 2, car_y)
            pt2 = (center_x + car_w // 2, car_y + car_h)
            cv2.rectangle(hud, pt1, pt2, color, -1)
            cv2.rectangle(hud, pt1, pt2, (255, 255, 255), 2)

    def _draw_telemetry(self, hud, speed_kmh, is_lane_departure):
        speed_str = f"{int(speed_kmh)}"
        
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(speed_str, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
        
        text_x = (self.width - text_size[0]) // 2
        text_y = self.height - 30
        
        cv2.putText(hud, speed_str, (text_x - 15, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, self.text_color, thickness)
        cv2.putText(hud, "km/h", (text_x + text_size[0] + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if is_lane_departure:
            cv2.putText(hud, "LDW", (20, self.height - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.hard_color, 2)

    def render(self, alert_level: str, is_lane_departure: bool, speed_kmh: float) -> np.ndarray:
        hud = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self._draw_background(hud)
        self._draw_lanes(hud, is_lane_departure)
        self._draw_vehicle_and_alerts(hud, alert_level)
        self._draw_telemetry(hud, speed_kmh, is_lane_departure)
        
        return hud