# type: ignore
import cv2
import math
import numpy as np
import time
import winsound
from beamngpy import BeamNGpy, Scenario, Vehicle
from lanedetector.lanedetector import LaneDetector, LaneDepartureDetector
from velocity_estimator import VelocityEstimator

MODEL_WEIGHTS_PATH = "lanedetector/weights/residual_unet_weights.pth.tar"
BEAMNG_PATH = "E:\\Games\\BeamNG.drive"
BEAMNG_USER = r"C:\Users\miker\AppData\Local\BeamNG.drive\0.32"

def init_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap


def beep_departure():
    try:
        winsound.Beep(2500, 200)
    except:
        pass


def draw_hud(image, left_state, right_state, left_edge, right_edge,
             speed_kph, flow_vis, mask_float):
    """
    Single combined HUD: lane overlay + departure bars + speed + flow minimap.
    Everything on one image, no extra windows.
    """
    h, w = image.shape[:2]
    result = image.copy()

    S_OK = LaneDepartureDetector.STATE_OK
    S_WARN = LaneDepartureDetector.STATE_WARNING
    S_DEPART = LaneDepartureDetector.STATE_DEPARTURE

    colors = {
        S_OK: (0, 200, 0),
        S_WARN: (0, 220, 255),
        S_DEPART: (0, 0, 255),
    }

    # --- Lane mask overlay (green tint on road) ---
    if mask_float is not None:
        overlay = result.copy()
        lane_mask = (mask_float > 0.5)
        overlay[lane_mask] = [0, 255, 0]
        result = cv2.addWeighted(result, 0.75, overlay, 0.25, 0)

    # --- Draw scan zone ---
    scan_y1 = int(h * 0.7)
    scan_y2 = int(h * 0.85)
    cv2.rectangle(result, (0, scan_y1), (w, scan_y2), (80, 80, 80), 1)

    # --- Draw detected lane edges ---
    if left_edge is not None:
        lx = int(left_edge * w)
        cv2.line(result, (lx, scan_y1), (lx, scan_y2), colors[left_state], 3)
    if right_edge is not None:
        rx = int(right_edge * w)
        cv2.line(result, (rx, scan_y1), (rx, scan_y2), colors[right_state], 3)

    # Center line
    cv2.line(result, (w // 2, scan_y1), (w // 2, scan_y2), (255, 255, 255), 1)

    # --- Side bars ---
    bar_w = 10
    bar_top = int(h * 0.25)
    bar_bot = int(h * 0.9)
    left_color = colors[left_state]
    right_color = colors[right_state]

    cv2.rectangle(result, (0, bar_top), (bar_w, bar_bot), left_color, -1)
    cv2.rectangle(result, (w - bar_w, bar_top), (w, bar_bot), right_color, -1)

    # --- Departure flash ---
    if left_state == S_DEPART:
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (int(w * 0.12), h), (0, 0, 180), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    if right_state == S_DEPART:
        overlay = result.copy()
        cv2.rectangle(overlay, (int(w * 0.88), 0), (w, h), (0, 0, 180), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

    # --- Top banner ---
    if left_state == S_DEPART or right_state == S_DEPART:
        text = "!! LANE DEPARTURE !!"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
        tx = (w - ts[0]) // 2
        ty = 45
        cv2.rectangle(result, (tx - 10, ty - 35), (tx + ts[0] + 10, ty + 8), (0, 0, 0), -1)
        cv2.putText(result, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        # Direction arrows
        if left_state == S_DEPART:
            pts = np.array([[50, h // 2 - 25], [15, h // 2], [50, h // 2 + 25]], np.int32)
            cv2.fillPoly(result, [pts], (0, 0, 255))
        if right_state == S_DEPART:
            pts = np.array([[w - 50, h // 2 - 25], [w - 15, h // 2], [w - 50, h // 2 + 25]], np.int32)
            cv2.fillPoly(result, [pts], (0, 0, 255))

    elif left_state == S_WARN or right_state == S_WARN:
        text = "~ Lane Warning ~"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = (w - ts[0]) // 2
        ty = 40
        cv2.rectangle(result, (tx - 10, ty - 28), (tx + ts[0] + 10, ty + 6), (0, 0, 0), -1)
        cv2.putText(result, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

    # --- Speed box (top right) ---
    cv2.rectangle(result, (w - 260, 8), (w - 10, 55), (0, 0, 0), -1)
    cv2.rectangle(result, (w - 260, 8), (w - 10, 55), (255, 255, 255), 1)
    cv2.putText(result, f"{speed_kph:.1f} km/h",
                (w - 250, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # --- Optical flow minimap (bottom right) ---
    if flow_vis is not None:
        fh, fw = flow_vis.shape[:2]
        mini_w = 200
        mini_h = int(mini_w * fh / fw)
        # Convert flow to HSV visualization
        mag, ang = cv2.cartToPolar(flow_vis[..., 0], flow_vis[..., 1])
        hsv = np.zeros((fh, fw, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flow_bgr = cv2.resize(flow_bgr, (mini_w, mini_h))

        # Place in bottom right corner
        y_off = h - mini_h - 10
        x_off = w - mini_w - 10
        result[y_off:y_off + mini_h, x_off:x_off + mini_w] = flow_bgr
        cv2.rectangle(result, (x_off - 1, y_off - 1),
                      (x_off + mini_w + 1, y_off + mini_h + 1), (255, 255, 255), 1)
        cv2.putText(result, "Optical Flow", (x_off, y_off - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- Edge distance readout (bottom left) ---
    state_names = {S_OK: "OK", S_WARN: "WARN", S_DEPART: "DEPART"}
    l_text = f"L: {left_edge:.2f}" if left_edge is not None else "L: --"
    r_text = f"R: {right_edge:.2f}" if right_edge is not None else "R: --"

    cv2.putText(result, f"{l_text}  [{state_names[left_state]}]",
                (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
    cv2.putText(result, f"{r_text}  [{state_names[right_state]}]",
                (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)

    return result


def start_simulation():
    bng = BeamNGpy('localhost', 64256, home=BEAMNG_PATH,
                   user=BEAMNG_USER)
    bng.open()

    scenario = Scenario('west_coast_usa', 'example')
    vehicle = Vehicle('ego_vehicle', model='etk800')
    scenario.add_vehicle(vehicle, pos=(-819.472, -500.348, 106.633),
                         rot_quat=(0.002, 0.004, 0.923, -0.386))

    scenario.make(bng)
    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.scenario.start()
    vehicle.ai.set_mode('disabled')

    # Spawn AI traffic
    bng.traffic.spawn(max_amount=5)

    return bng, scenario, vehicle


def main():
    print("Starting simulation...")
    bng, scenario, vehicle = start_simulation()

    print("Loading AI lane detector...")
    ld = LaneDetector(checkpoint_path=MODEL_WEIGHTS_PATH)
    ldd = LaneDepartureDetector()
    vel = VelocityEstimator()
    print("All systems loaded.")

    input("Press [Enter] to begin...")
    cap = init_camera()

    last_beep = 0
    beep_cooldown = 1.5
    paused = False
    frame_count = 0
    fps_time = time.time()
    display_fps = 0.0

    print("Running. Keys: q=quit  p=pause")

    while True:
        vehicle.sensors.poll()

        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        # --- Lane segmentation ---
        mask_float, mask_binary = ld.detect(frame)

        # --- Lane departure from mask directly ---
        left_state, right_state, left_edge, right_edge = ldd.update(mask_binary)

        # --- Velocity estimation ---
        speed_kph, flow = vel.update(frame)

        # --- Beep on departure ---
        now = time.time()
        if (left_state == LaneDepartureDetector.STATE_DEPARTURE or
                right_state == LaneDepartureDetector.STATE_DEPARTURE):
            if (now - last_beep) > beep_cooldown:
                beep_departure()
                last_beep = now

        # --- Single combined HUD ---
        display = draw_hud(frame, left_state, right_state, left_edge, right_edge,
                           speed_kph, flow, mask_float)

        # FPS counter
        frame_count += 1
        elapsed = now - fps_time
        if elapsed >= 1.0:
            display_fps = frame_count / elapsed
            frame_count = 0
            fps_time = now

        cv2.putText(display, f"FPS: {display_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("ADAS", display)

        # Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
            while paused:
                vehicle.control(steering=0, throttle=0, brake=1.0)
                k = cv2.waitKey(30) & 0xFF
                if k == ord('p'):
                    paused = False
                    print("RESUMED")
                elif k == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    bng.traffic.stop()
                    bng.close()
                    return

    cap.release()
    cv2.destroyAllWindows()
    bng.traffic.stop()
    bng.close()


if __name__ == "__main__":
    main()