import threading
import time

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

class AudioManager:
    def __init__(self):
        self.last_beep_time = 0.0

    def play_beep(self, freq: int, duration_ms: int, cooldown_s: float):
        now = time.time()
        if now - self.last_beep_time > cooldown_s:
            self.last_beep_time = now
            if _HAS_WINSOUND:
                threading.Thread(
                    target=winsound.Beep,
                    args=(int(freq), int(duration_ms)),
                    daemon=True
                ).start()
            else:
                print(f"ALERT: {freq}Hz for {duration_ms}ms")