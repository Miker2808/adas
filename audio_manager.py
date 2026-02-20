import os
import threading
import time

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

class AudioManager:
    def __init__(self):
        self.last_hard_time = 0.0
        self.last_soft_time = 0.0
        self.last_lane_time = 0.0
        
        if _HAS_PYGAME:
            pygame.mixer.init()
            self.sound_hard = pygame.mixer.Sound("sounds/hard_warning.wav") if os.path.exists("sounds/hard_warning.wav") else None
            self.sound_soft = pygame.mixer.Sound("sounds/soft_warning.wav") if os.path.exists("sounds/soft_warning.wav") else None
            self.sound_lane = pygame.mixer.Sound("sounds/lane_departure.wav") if os.path.exists("sounds/lane_departure.wav") else None
        else:
            self.sound_hard = None
            self.sound_soft = None
            self.sound_lane = None

    def play_hard_warning(self, freq: int, duration_ms: int, cooldown_s: float):
        now = time.time()
        if now - self.last_hard_time > cooldown_s:
            self.last_hard_time = now
            if _HAS_PYGAME and self.sound_hard:
                self.sound_hard.play()
            elif _HAS_WINSOUND:
                threading.Thread(
                    target=winsound.Beep,
                    args=(int(freq), int(duration_ms)),
                    daemon=True
                ).start()
            else:
                print(f"HARD WARNING ALERT")

    def play_soft_warning(self, freq: int, duration_ms: int, cooldown_s: float):
        now = time.time()
        if now - self.last_soft_time > cooldown_s:
            self.last_soft_time = now
            if _HAS_PYGAME and self.sound_soft:
                self.sound_soft.play()
            elif _HAS_WINSOUND:
                threading.Thread(
                    target=winsound.Beep,
                    args=(int(freq), int(duration_ms)),
                    daemon=True
                ).start()
            else:
                print(f"SOFT WARNING ALERT")

    def play_lane_warning(self, freq: int, duration_ms: int, cooldown_s: float):
        now = time.time()
        if now - self.last_lane_time > cooldown_s:
            self.last_lane_time = now
            if _HAS_PYGAME and self.sound_lane:
                self.sound_lane.play()
            elif _HAS_WINSOUND:
                threading.Thread(
                    target=winsound.Beep,
                    args=(int(freq), int(duration_ms)),
                    daemon=True
                ).start()
            else:
                print(f"LANE DEPARTURE ALERT")