# Author: Akash Arakere Mata 
import json, math
from typing import Tuple

import numpy as np
import cv2
from scipy.spatial import cKDTree

LIDAR_JSON_PATH = r"D:\slam_27\static_map\lidar_scans.json"  

SHOW_LIVE       = True
SHOW_EVERY_N    = 50      
WINDOW_NAME     = "Occupancy Map (Live)"
SCALE_VIEW      = 2        
DRAW_TRAJ_LIVE  = True     

RES_M_PER_CELL = 0.05     # 5 cm per cell
MAP_SIZE_M     = 30.0     # 30m x 30m
LOG_FREE = -0.35
LOG_OCC  = +0.85
LOG_MIN  = -4.0
LOG_MAX  = +4.0

FREE_TH = 0.40   
OCC_TH  = 0.70   

RANGE_MIN_M = 0.05
RANGE_MAX_M = 12.0
BEAM_STEP   = 4         

COARSE_DXY   = 0.45
COARSE_DTH   = math.radians(10)
COARSE_STEP  = 0.05
COARSE_TSTEP = math.radians(1.5)

FINE_DXY     = 0.15
FINE_DTH     = math.radians(3)
FINE_STEP    = 0.02
FINE_TSTEP   = math.radians(0.5)

ICP_ITERS      = 6        
ICP_MAX_DIST_M = 0.35
ICP_SAMPLE_MAP = 4000     

MIN_OCC_CELLS_FOR_MATCH = 200

def load_scans(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def world_to_grid(x_m: float, y_m: float, origin_xy_m: Tuple[float, float], res: float) -> Tuple[int, int]:
    ox, oy = origin_xy_m
    gx = int((x_m - ox) / res)
    gy = int((y_m - oy) / res)
    return gx, gy

def grid_to_world(gx: int, gy: int, origin_xy_m: Tuple[float, float], res: float) -> Tuple[float, float]:
    ox, oy = origin_xy_m
    x = ox + (gx + 0.5) * res
    y = oy + (gy + 0.5) * res
    return x, y

def bresenham_free_cells(x0, y0, x1, y1):
    """Cells along line excluding endpoint."""
    points = []
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        points.append((x, y))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    if points:
        points.pop()
    return points

def scan_to_points(scan, step=2, rmin=0.05, rmax=12.0):
    amin = float(scan["angle_min_rad"])
    ainc = float(scan["angle_increment_rad"])
    ranges = scan["ranges_m"]
    pts = []
    for i in range(0, len(ranges), step):
        r = ranges[i]
        if r is None:
            continue
        try:
            r = float(r)
        except:
            continue
        if not np.isfinite(r) or r < rmin or r > rmax:
            continue
        a = amin + i * ainc
        pts.append((r * math.cos(a), r * math.sin(a)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)

def transform_points(pts_xy: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s,  c]], dtype=np.float32)
    out = pts_xy @ R.T
    out[:, 0] += x
    out[:, 1] += y
    return out

def make_distance_transform(occ_mask: np.ndarray) -> np.ndarray:
    inv = (1 - occ_mask).astype(np.uint8)
    return cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)

def score_pose_with_dt(dt: np.ndarray, pts_world: np.ndarray, origin_xy_m, res: float) -> float:
    H, W = dt.shape
    gx = ((pts_world[:, 0] - origin_xy_m[0]) / res).astype(np.int32)
    gy = ((pts_world[:, 1] - origin_xy_m[1]) / res).astype(np.int32)
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    if not np.any(ok):
        return 1e9
    d = dt[gy[ok], gx[ok]]
    return float(np.mean(np.clip(d, 0, 50)))

def correlative_match(prev_pose, scan_pts, dt, origin_xy_m, res,
                      dxy, dth, step_xy, step_th):
    best_pose = prev_pose
    best_score = 1e18

    dxs = np.arange(-dxy, dxy + 1e-9, step_xy)
    dys = np.arange(-dxy, dxy + 1e-9, step_xy)
    dts = np.arange(-dth, dth + 1e-12, step_th)

    px, py, pth = prev_pose
    for dtheta in dts:
        th = wrap_angle(pth + dtheta)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        rot = scan_pts @ R.T

        for dx in dxs:
            x = px + dx
            for dy in dys:
                y = py + dy
                pts_w = rot.copy()
                pts_w[:, 0] += x
                pts_w[:, 1] += y
                sc = score_pose_with_dt(dt, pts_w, origin_xy_m, res)
                if sc < best_score:
                    best_score = sc
                    best_pose = (x, y, th)
    return best_pose, best_score

def extract_map_points(occ_mask: np.ndarray, origin_xy_m, res: float, pose, radius_m: float = 6.0) -> np.ndarray:
    H, W = occ_mask.shape
    px, py, _ = pose
    cx, cy = world_to_grid(px, py, origin_xy_m, res)
    r = int(radius_m / res)
    x0 = max(0, cx - r); x1 = min(W, cx + r)
    y0 = max(0, cy - r); y1 = min(H, cy + r)

    sub = occ_mask[y0:y1, x0:x1]
    ys, xs = np.where(sub > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    gx = xs + x0
    gy = ys + y0

    if len(gx) > ICP_SAMPLE_MAP:
        idx = np.random.choice(len(gx), ICP_SAMPLE_MAP, replace=False)
        gx = gx[idx]; gy = gy[idx]

    pts = np.zeros((len(gx), 2), dtype=np.float32)
    for i in range(len(gx)):
        pts[i] = grid_to_world(int(gx[i]), int(gy[i]), origin_xy_m, res)
    return pts

def icp_refine(pose_init, scan_pts_sensor, map_pts_world, iters=8, max_dist=0.35):
    if len(map_pts_world) < 50 or len(scan_pts_sensor) < 20:
        return pose_init

    pose = list(pose_init)
    tree = cKDTree(map_pts_world)

    for _ in range(iters):
        scan_world = transform_points(scan_pts_sensor, tuple(pose))
        dists, idx = tree.query(scan_world, k=1, workers=-1)
        ok = dists < max_dist
        if np.count_nonzero(ok) < 20:
            break

        src = scan_world[ok]
        tgt = map_pts_world[idx[ok]]

        src_mean = src.mean(axis=0)
        tgt_mean = tgt.mean(axis=0)
        X = src - src_mean
        Y = tgt - tgt_mean

        Hm = X.T @ Y
        U, _, Vt = np.linalg.svd(Hm)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T
        t = tgt_mean - (R @ src_mean)

        dtheta = math.atan2(R[1, 0], R[0, 0])

        pose[0] += float(t[0])
        pose[1] += float(t[1])
        pose[2] = wrap_angle(pose[2] + dtheta)

    return tuple(pose)

def integrate_scan(logodds: np.ndarray, pose, scan_pts_sensor, origin_xy_m, res):
    H, W = logodds.shape
    rx, ry, _ = pose
    r_gx, r_gy = world_to_grid(rx, ry, origin_xy_m, res)
    if not (0 <= r_gx < W and 0 <= r_gy < H):
        return

    endpoints = transform_points(scan_pts_sensor, pose)

    for p in endpoints:
        gx, gy = world_to_grid(float(p[0]), float(p[1]), origin_xy_m, res)
        if gx < 0 or gx >= W or gy < 0 or gy >= H:
            continue

        free_cells = bresenham_free_cells(r_gx, r_gy, gx, gy)
        for fx, fy in free_cells[::2]:
            if 0 <= fx < W and 0 <= fy < H:
                logodds[fy, fx] = np.clip(logodds[fy, fx] + LOG_FREE, LOG_MIN, LOG_MAX)

        logodds[gy, gx] = np.clip(logodds[gy, gx] + LOG_OCC, LOG_MIN, LOG_MAX)

def render_from_logodds(logodds: np.ndarray, L_FREE_TH: float, L_OCC_TH: float) -> np.ndarray:
    """Fast render without exp(): Gray unknown, White free, Black occupied."""
    img = np.full(logodds.shape, 160, dtype=np.uint8)
    img[logodds < L_FREE_TH] = 255
    img[logodds > L_OCC_TH]  = 0
    return img


def main():
    scans = load_scans(LIDAR_JSON_PATH)
    print(f"Loaded {len(scans)} scans from JSON")

    grid_w = int(MAP_SIZE_M / RES_M_PER_CELL)
    grid_h = int(MAP_SIZE_M / RES_M_PER_CELL)
    origin_xy_m = (-MAP_SIZE_M / 2.0, -MAP_SIZE_M / 2.0)

    logodds = np.zeros((grid_h, grid_w), dtype=np.float32)

    # Convert probability thresholds ONCE to log-odds thresholds (big speedup)
    # L = log(p/(1-p))
    L_FREE_TH = float(math.log(FREE_TH / (1.0 - FREE_TH)))
    L_OCC_TH  = float(math.log(OCC_TH  / (1.0 - OCC_TH)))

    pose = (0.0, 0.0, 0.0)
    trajectory = [pose]

    if SHOW_LIVE:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for k, scan in enumerate(scans):
        scan_pts = scan_to_points(scan, step=BEAM_STEP, rmin=RANGE_MIN_M, rmax=RANGE_MAX_M)
        if len(scan_pts) < 30:
            continue

        # Build occ_mask directly from logodds (NO exp())
        occ_mask = (logodds > L_OCC_TH).astype(np.uint8)
        occ_count = int(occ_mask.sum())

        if occ_count >= MIN_OCC_CELLS_FOR_MATCH:
            dt = make_distance_transform(occ_mask)

            pose_coarse, _ = correlative_match(
                pose, scan_pts, dt, origin_xy_m, RES_M_PER_CELL,
                COARSE_DXY, COARSE_DTH, COARSE_STEP, COARSE_TSTEP
            )
            pose_fine, _ = correlative_match(
                pose_coarse, scan_pts, dt, origin_xy_m, RES_M_PER_CELL,
                FINE_DXY, FINE_DTH, FINE_STEP, FINE_TSTEP
            )

            map_pts = extract_map_points(occ_mask, origin_xy_m, RES_M_PER_CELL, pose_fine, radius_m=6.0)
            pose = icp_refine(pose_fine, scan_pts, map_pts, iters=ICP_ITERS, max_dist=ICP_MAX_DIST_M)

        integrate_scan(logodds, pose, scan_pts, origin_xy_m, RES_M_PER_CELL)
        trajectory.append(pose)

        if (k + 1) % 50 == 0:
            print(f"Processed {k+1}/{len(scans)} scans | occ={occ_count} | pose=({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.1f}deg)")

        # ---- Live display  ----
        if SHOW_LIVE and ((k + 1) % SHOW_EVERY_N == 0):
            base = render_from_logodds(logodds, L_FREE_TH, L_OCC_TH)

            if DRAW_TRAJ_LIVE:
                view_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

                # trajectory 
                traj_px = []
                for (x, y, _) in trajectory:
                    gx, gy = world_to_grid(x, y, origin_xy_m, RES_M_PER_CELL)
                    traj_px.append((gx, gy))
                for i in range(1, len(traj_px)):
                    cv2.line(view_bgr, traj_px[i-1], traj_px[i], (255, 0, 0), 2)

                if traj_px:
                    cv2.circle(view_bgr, traj_px[0], 4, (0, 255, 0), -1)   # start green
                    cv2.circle(view_bgr, traj_px[-1], 4, (0, 0, 255), -1)  # end red

                show_img = view_bgr
            else:
                show_img = base

            if SCALE_VIEW != 1:
                show_img = cv2.resize(
                    show_img,
                    (show_img.shape[1] * SCALE_VIEW, show_img.shape[0] * SCALE_VIEW),
                    interpolation=cv2.INTER_NEAREST
                )

            cv2.imshow(WINDOW_NAME, show_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopped early by user (q).")
                break

   
    final_img = render_from_logodds(logodds, L_FREE_TH, L_OCC_TH)
    out_map = "static_map.png"
    cv2.imwrite(out_map, final_img)
    print(f"Saved final map (no trajectory/dots): {out_map}")

    if SHOW_LIVE:
        show_final = final_img
        if SCALE_VIEW != 1:
            show_final = cv2.resize(
                show_final,
                (show_final.shape[1] * SCALE_VIEW, show_final.shape[0] * SCALE_VIEW),
                interpolation=cv2.INTER_NEAREST
            )
        cv2.imshow(WINDOW_NAME, show_final)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
