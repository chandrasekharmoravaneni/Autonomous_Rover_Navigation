import os, json, math, time, base64, threading, random
from collections import deque

import numpy as np
import cv2
import paho.mqtt.client as mqtt

# MQTT CONFIG
MQTT_HOST = os.environ.get("MQTT_HOST", "195.37.48.238")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))

TOPIC_POSE_IN = os.environ.get("TOPIC_POSE_IN", "SLAM/pose")
TOPIC_SCAN_IN = os.environ.get("TOPIC_SCAN_IN", "lidar/scans")

TOPIC_MAP_OUT      = os.environ.get("TOPIC_MAP_OUT", "slam/live")
TOPIC_POSE_EST_OUT = os.environ.get("TOPIC_POSE_EST_OUT", "slam/pose_est")

# MAP CONFIG
RES_M_PER_CELL = 0.05
MAP_SIZE_M = 30.0

LOG_FREE = -0.35
LOG_OCC  = +0.85
LOG_MIN  = -4.0
LOG_MAX  = +4.0

FREE_TH = 0.40
OCC_TH  = 0.70
# LIDAR CONFIG

RANGE_MIN_M = 0.05
RANGE_MAX_M = 12.0
BEAM_STEP_MAP = 2
BEAM_STEP_LIK = 8

REVERSE_SCAN_ORDER = False
SYNC_MAX_DT = 0.08  # seconds

def _angles_are_huge(angle_min: float, angle_inc: float) -> bool:
    return (abs(angle_min) > 10.0) or (abs(angle_inc) > 1.0)

# PF CONFIG
N_PARTICLES = 250

SIGMA_DXY_BODY = 0.02
SIGMA_DTH      = math.radians(1.5)

SIGMA_XY_INIT = 0.05
SIGMA_TH_INIT = math.radians(3.0)

RESAMPLE_NEFF_RATIO = 0.55

# Publish throttles (IMPORTANT for stability)
MAP_PUB_HZ  = float(os.environ.get("MAP_PUB_HZ", "2.0"))   
POSE_PUB_HZ = float(os.environ.get("POSE_PUB_HZ", "20.0")) 

MAP_PUB_QOS   = int(os.environ.get("MAP_PUB_QOS", "0"))
MAP_PUB_RETAIN = (os.environ.get("MAP_PUB_RETAIN", "1") != "0")

# OPTIONAL LOCAL VIS
SHOW_LOCAL = True
WIN_NAME = "LIVE SLAM (PF + Map)"
DRAW_PARTICLES = True
PARTICLE_DRAW_STRIDE = 3
PARTICLE_COLOR = (0, 255, 0)  # BGR
PARTICLE_RADIUS = 2
DRAW_BEST_PARTICLE = True
BEST_COLOR = (255, 0, 255)
BEST_RADIUS = 5
DRAW_EST_POSE = True

# Shared state (buffered for time alignment)
lock = threading.Lock()
pose_buf = deque(maxlen=500)   
scan_buf = deque(maxlen=200)   

# Helpers
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            yield x, y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            yield x, y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    yield x1, y1

# Map (log-odds occupancy grid)
class LogOddsMap:
    def __init__(self, size_m: float, res: float):
        self.res = float(res)
        n = int(round(size_m / res))
        if n % 2 == 1:
            n += 1
        self.n = n
        self.grid = np.zeros((n, n), dtype=np.float32)
        self.cx = n // 2
        self.cy = n // 2

        # bottom-left of the map in world meters (centered map)
        self.origin_x_m = -size_m / 2.0
        self.origin_y_m = -size_m / 2.0

    def world_to_cell(self, x_m: float, y_m: float):
        ix = int(round(x_m / self.res)) + self.cx
        iy = int(round(y_m / self.res)) + self.cy
        return ix, iy

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.n and 0 <= iy < self.n

    def update_ray(self, x0: int, y0: int, x1: int, y1: int):
        pts = list(bresenham(x0, y0, x1, y1))
        if len(pts) < 2:
            return
        for (x, y) in pts[:-1]:
            if self.in_bounds(x, y):
                self.grid[y, x] = clamp(self.grid[y, x] + LOG_FREE, LOG_MIN, LOG_MAX)
        xe, ye = pts[-1]
        if self.in_bounds(xe, ye):
            self.grid[ye, xe] = clamp(self.grid[ye, xe] + LOG_OCC, LOG_MIN, LOG_MAX)

    def prob(self) -> np.ndarray:
        odds = np.exp(self.grid)
        return odds / (1.0 + odds)

    def p_occ_cell(self, ix: int, iy: int) -> float:
        l = float(self.grid[iy, ix])
        return 1.0 / (1.0 + math.exp(-l))

    def to_occ_image_u8(self) -> np.ndarray:
        p = self.prob()
        occ = np.full(p.shape, 128, dtype=np.uint8)   # unknown gray
        occ[p < FREE_TH] = 255                        # free white
        occ[p > OCC_TH]  = 0                          # occupied black
        return occ

# Particle Filter
class Particle:
    __slots__ = ("x", "y", "th", "w")
    def __init__(self, x, y, th, w):
        self.x = x
        self.y = y
        self.th = th
        self.w = w

class ParticleFilter:
    def __init__(self, x0, y0, th0):
        self.particles = [
            Particle(
                x0 + random.gauss(0, SIGMA_XY_INIT),
                y0 + random.gauss(0, SIGMA_XY_INIT),
                wrap_pi(th0 + random.gauss(0, SIGMA_TH_INIT)),
                1.0 / N_PARTICLES
            )
            for _ in range(N_PARTICLES)
        ]

    def predict_body_delta(self, dx_body: float, dy_body: float, dth: float):
        for p in self.particles:
            ndx = dx_body + random.gauss(0, SIGMA_DXY_BODY)
            ndy = dy_body + random.gauss(0, SIGMA_DXY_BODY)
            ndt = dth     + random.gauss(0, SIGMA_DTH)

            c = math.cos(p.th)
            s = math.sin(p.th)
            p.x += c * ndx - s * ndy
            p.y += s * ndx + c * ndy
            p.th = wrap_pi(p.th + ndt)

    def update_weights(self, lomap: LogOddsMap, angle_min: float, angle_inc: float, ranges: np.ndarray):
        n = int(ranges.size)
        if n < 5:
            return

        idxs = range(0, n, BEAM_STEP_LIK)
        logw = np.zeros(len(self.particles), dtype=np.float64)

        for i, p in enumerate(self.particles):
            ssum = 0.0
            for k in idxs:
                r = float(ranges[k])
                if not np.isfinite(r) or r < RANGE_MIN_M:
                    continue
                if r > RANGE_MAX_M:
                    r = RANGE_MAX_M

                a = angle_min + k * angle_inc
                g = p.th + a
                ex = p.x + r * math.cos(g)
                ey = p.y + r * math.sin(g)

                ix, iy = lomap.world_to_cell(ex, ey)
                if lomap.in_bounds(ix, iy):
                    pocc = lomap.p_occ_cell(ix, iy)
                    ssum += math.log(1e-6 + (0.2 + 0.8 * pocc))
                else:
                    ssum += math.log(0.5)

            logw[i] = ssum

        m = float(np.max(logw))
        w = np.exp(logw - m)
        wsum = float(np.sum(w)) + 1e-12
        w /= wsum

        for i, p in enumerate(self.particles):
            p.w = float(w[i])

    def neff(self) -> float:
        w = np.array([p.w for p in self.particles], dtype=np.float64)
        return 1.0 / (np.sum(w * w) + 1e-12)

    def resample(self):
        weights = np.array([p.w for p in self.particles], dtype=np.float64)
        weights /= (weights.sum() + 1e-12)

        n = len(self.particles)
        positions = (np.arange(n) + random.random()) / n
        indexes = np.zeros(n, dtype=np.int32)

        cumsum = np.cumsum(weights)
        i = j = 0
        while i < n:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        newp = []
        for idx in indexes:
            p = self.particles[int(idx)]
            newp.append(Particle(p.x, p.y, p.th, 1.0 / n))
        self.particles = newp

    def mean_pose(self):
        xs = np.array([p.x for p in self.particles], dtype=np.float64)
        ys = np.array([p.y for p in self.particles], dtype=np.float64)
        ths = np.array([p.th for p in self.particles], dtype=np.float64)
        ws  = np.array([p.w for p in self.particles], dtype=np.float64)
        ws /= (ws.sum() + 1e-12)

        x = float((xs * ws).sum())
        y = float((ys * ws).sum())
        c = float((np.cos(ths) * ws).sum())
        s = float((np.sin(ths) * ws).sum())
        th = math.atan2(s, c)
        return x, y, th

    def best_particle(self):
        return max(self.particles, key=lambda p: p.w)
# Parsing MQTT messages

def parse_pose_msg(msg: dict):
    t = float(msg["timestamp"]["t_mono"])
    x = float(msg["pose"]["x_m"])
    y = float(msg["pose"]["y_m"])
    th = wrap_pi(float(msg["pose"]["theta_rad"]))
    return t, x, y, th

def parse_scan_msg(msg: dict):
    t = float(msg["t_mono"])
    angle_min = float(msg["angle_min_rad"])
    angle_inc = float(msg["angle_increment_rad"])
    ranges = np.asarray(msg["ranges_m"], dtype=np.float32)

    if REVERSE_SCAN_ORDER:
        ranges = ranges[::-1]

    if _angles_are_huge(angle_min, angle_inc):
        n = int(ranges.size)
        angles = angle_min + angle_inc * np.arange(n, dtype=np.float64)
        center = float(angles[n // 2])
        angle_min = wrap_pi(angle_min - center)

    return t, angle_min, angle_inc, ranges

# Robust time alignment
def pop_time_aligned_pair():
    with lock:
        if len(scan_buf) == 0 or len(pose_buf) == 0:
            return None, None

        st, s_angle_min, s_angle_inc, s_ranges = scan_buf[0]

        best_i = None
        best_dt = 1e9
        for i, (pt, px, py, pth) in enumerate(pose_buf):
            dt = abs(pt - st)
            if dt < best_dt:
                best_dt = dt
                best_i = i

        if best_i is None or best_dt > SYNC_MAX_DT:
            newest_pose_t = pose_buf[-1][0]

            if st < newest_pose_t - SYNC_MAX_DT:
                scan_buf.popleft()
            else:
                pose_buf.popleft()

            while len(pose_buf) > 0 and len(scan_buf) > 0:
                if pose_buf[0][0] < scan_buf[0][0] - 2.0 * SYNC_MAX_DT:
                    pose_buf.popleft()
                else:
                    break
            return None, None

        for _ in range(best_i):
            pose_buf.popleft()

        pose = pose_buf[0]
        scan = scan_buf.popleft()
        return pose, scan
# Publish map & pose
def publish_map(client: mqtt.Client, lomap: LogOddsMap, t_mono_s: float):
    occ = lomap.to_occ_image_u8()
    occ = np.ascontiguousarray(np.asarray(occ, dtype=np.uint8))

    ok, buf = cv2.imencode(".png", occ)
    if not ok:
        print("[WARN] publish_map: cv2.imencode failed")
        return

    payload = {
        "encoding": "png_base64",
        "data": base64.b64encode(buf.tobytes()).decode("ascii"),
        "resolution_m_per_px": float(lomap.res),
        "origin_x_m": float(lomap.origin_x_m),
        "origin_y_m": float(lomap.origin_y_m),
        "width_px": int(lomap.n),
        "height_px": int(lomap.n),
        "timestamp_ms": int(t_mono_s * 1000.0)
    }
    try:
        client.publish(TOPIC_MAP_OUT, json.dumps(payload), qos=MAP_PUB_QOS, retain=MAP_PUB_RETAIN)
    except Exception as e:
        print(f"[WARN] publish_map: publish failed: {e}")

def publish_pose_est(client: mqtt.Client, lomap: LogOddsMap, x, y, th, t_mono_s: float, neff: float):
    # Convert centered (4-quadrant) -> bottom-left (2-quadrant)
    x_bl = float(x - lomap.origin_x_m)
    y_bl = float(y - lomap.origin_y_m)

    payload = {
        "x_m": x_bl,
        "y_m": y_bl,
        "theta_rad": float(th),
        "timestamp_ms": int(t_mono_s * 1000.0),
        "pose_valid": True,
        "neff": float(neff),
        "frame": "bottom_left_0_0"
    }
    try:
        client.publish(TOPIC_POSE_EST_OUT, json.dumps(payload), qos=0, retain=False)
    except Exception as e:
        print(f"[WARN] publish_pose_est: publish failed: {e}")

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected rc={rc} to {MQTT_HOST}:{MQTT_PORT}")
    client.subscribe(TOPIC_POSE_IN, qos=0)
    client.subscribe(TOPIC_SCAN_IN, qos=0)
    print(f"[MQTT] Subscribed: {TOPIC_POSE_IN}, {TOPIC_SCAN_IN}")

def on_disconnect(client, userdata, rc):
    print(f"[MQTT] Disconnected rc={rc}")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8", errors="ignore"))
    except Exception:
        return

    if msg.topic == TOPIC_POSE_IN:
        try:
            pose = parse_pose_msg(data)
        except Exception:
            return
        with lock:
            pose_buf.append(pose)

    elif msg.topic == TOPIC_SCAN_IN:
        try:
            scan = parse_scan_msg(data)
        except Exception:
            return
        with lock:
            scan_buf.append(scan)
# Main
def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    client.reconnect_delay_set(min_delay=1, max_delay=10)
    client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    lomap = LogOddsMap(MAP_SIZE_M, RES_M_PER_CELL)

    pf = None
    last_pose = None  # (t,x,y,th)

    if SHOW_LOCAL:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    last_map_pub_t  = 0.0
    last_pose_pub_t = 0.0

    try:
        while True:
            pose, scan = pop_time_aligned_pair()
            if pose is None or scan is None:
                time.sleep(0.002)
                continue

            tp, x, y, th = pose
            ts, angle_min, angle_inc, ranges = scan

            if pf is None:
                pf = ParticleFilter(x, y, th)
                last_pose = (tp, x, y, th)
                continue

            _, lx, ly, lth = last_pose
            dx_w = x - lx
            dy_w = y - ly
            dth  = wrap_pi(th - lth)
            last_pose = (tp, x, y, th)

            c = math.cos(-lth)
            s = math.sin(-lth)
            dx_body = c * dx_w - s * dy_w
            dy_body = s * dx_w + c * dy_w

            pf.predict_body_delta(dx_body, dy_body, dth)
            pf.update_weights(lomap, angle_min, angle_inc, ranges)

            neff = pf.neff()
            if neff < RESAMPLE_NEFF_RATIO * N_PARTICLES:
                pf.resample()

            bx, by, bth = pf.mean_pose()

            # Map update
            x0c, y0c = lomap.world_to_cell(bx, by)
            if lomap.in_bounds(x0c, y0c):
                n = int(ranges.size)
                for i in range(0, n, BEAM_STEP_MAP):
                    r = float(ranges[i])
                    if not np.isfinite(r) or r < RANGE_MIN_M:
                        continue
                    if r > RANGE_MAX_M:
                        r = RANGE_MAX_M

                    a = angle_min + i * angle_inc
                    g = bth + a
                    ex = bx + r * math.cos(g)
                    ey = by + r * math.sin(g)
                    x1c, y1c = lomap.world_to_cell(ex, ey)
                    if lomap.in_bounds(x1c, y1c):
                        lomap.update_ray(x0c, y0c, x1c, y1c)

            # Throttled publishing
            if MAP_PUB_HZ > 0:
                if (ts - last_map_pub_t) >= (1.0 / MAP_PUB_HZ):
                    publish_map(client, lomap, ts)
                    last_map_pub_t = ts

            if POSE_PUB_HZ > 0:
                if (ts - last_pose_pub_t) >= (1.0 / POSE_PUB_HZ):
                    publish_pose_est(client, lomap, bx, by, bth, ts, neff)
                    last_pose_pub_t = ts

            # Local display
            if SHOW_LOCAL:
                occ = lomap.to_occ_image_u8()
                occ = np.ascontiguousarray(np.asarray(occ, dtype=np.uint8))
                img = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)

                if DRAW_PARTICLES and pf is not None:
                    for p in pf.particles[::max(1, PARTICLE_DRAW_STRIDE)]:
                        px, py = lomap.world_to_cell(p.x, p.y)
                        if lomap.in_bounds(px, py):
                            cv2.circle(img, (px, py), PARTICLE_RADIUS, PARTICLE_COLOR, -1)

                if DRAW_BEST_PARTICLE and pf is not None:
                    bp = pf.best_particle()
                    bpx, bpy = lomap.world_to_cell(bp.x, bp.y)
                    if lomap.in_bounds(bpx, bpy):
                        cv2.circle(img, (bpx, bpy), BEST_RADIUS, BEST_COLOR, -1)

                if DRAW_EST_POSE:
                    cx, cy = lomap.world_to_cell(bx, by)
                    if lomap.in_bounds(cx, cy):
                        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

                        
                        # ONLY CHANGE: rotate displayed heading line by +90 degrees
                        disp_th = wrap_pi(bth + (math.pi / 2.0))

                        hx = int(round(cx + 20 * math.cos(disp_th)))
                        hy = int(round(cy + 20 * math.sin(disp_th)))
                        cv2.line(img, (cx, cy), (hx, hy), (255, 0, 0), 2)

                cv2.putText(
                    img,
                    f"t={ts:.3f}  est=({bx:.2f},{by:.2f},{math.degrees(bth):.1f}deg)  Neff={neff:.1f}  map={MAP_PUB_HZ:.1f}Hz",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(WIN_NAME, img)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

    finally:
        client.loop_stop()
        try:
            client.disconnect()
        except Exception:
            pass
        if SHOW_LOCAL:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()