#!/usr/bin/env python3
"""
Using Imu to get position and orientation (INS) and self localization
"""

import signal
import time
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R

from ahrs.filters import Madgwick

from sbp.client.drivers.network_drivers import TCPDriver
from sbp.client import Framer
from sbp.imu import MsgImuRaw, MsgImuAux

# ================= CONFIG =================
IP = "195.37.48.233"
PORT = 55555

G = 9.80665
BIAS_SAMPLES = 300

ZUPT_ACC = 0.15      # m/s^2
ZUPT_GYRO = 1.0      # deg/s

CSV_FILE = "ins_output.csv"

PRINT_INTERVAL = 0.2   # seconds between console prints

# ================= STATE =================
running = True
acc_range_g = None
gyro_range_dps = None

bias_buf = []
acc_bias = np.zeros(3)
gyro_bias = np.zeros(3)

# Madgwick filter
madgwick = Madgwick(beta=0.1)
quat = np.array([1.0, 0.0, 0.0, 0.0])

vel = np.zeros(3)
pos = np.zeros(3)

last_time = None
last_print_time = 0.0

# ================= SIGNAL =================
def stop_handler(sig, frame):
    global running
    print("\nStopping...\n")
    running = False

signal.signal(signal.SIGINT, stop_handler)

# ================= IMU AUX =================
def decode_imu_conf(imu_conf):
    acc_code = imu_conf & 0x0F
    gyro_code = (imu_conf >> 4) & 0x0F

    acc_ranges = {0:2, 1:4, 2:8, 3:16}
    gyro_ranges = {0:2000, 1:1000, 2:500, 3:250, 4:125}

    return acc_ranges.get(acc_code), gyro_ranges.get(gyro_code)

# ================= MAIN =================
def main():
    global acc_range_g, gyro_range_dps
    global acc_bias, gyro_bias, last_time
    global quat, vel, pos, last_print_time

    print(f"Connecting to {IP}:{PORT}")
    driver = TCPDriver(IP, PORT, timeout=5, reconnect=True)
    framer = Framer(driver.read, driver.write)

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "z", "yaw_deg"])

        print("Waiting for IMU_AUX and collecting bias (KEEP IMU STILL)...")

        for msg, meta in framer:

            if not running:
                break

            now = time.time()
            if last_time is None:
                last_time = now
                continue

            dt = now - last_time
            last_time = now
            if dt <= 0 or dt > 0.1:
                continue

            # ---------- IMU_AUX ----------
            if isinstance(msg, MsgImuAux):
                if acc_range_g is None:
                    acc_range_g, gyro_range_dps = decode_imu_conf(msg.imu_conf)
                    print(f"IMU ranges locked: ±{acc_range_g} g, ±{gyro_range_dps} dps")
                continue

            # ---------- IMU_RAW ----------
            if not isinstance(msg, MsgImuRaw):
                continue
            if acc_range_g is None:
                continue

            acc_scale = acc_range_g * G / 32768.0
            gyro_scale = gyro_range_dps / 32768.0

            acc = np.array([
                msg.acc_x * acc_scale,
                msg.acc_y * acc_scale,
                msg.acc_z * acc_scale
            ])

            gyro = np.array([
                msg.gyr_x * gyro_scale,
                msg.gyr_y * gyro_scale,
                msg.gyr_z * gyro_scale
            ])  # deg/s

            # ---------- BIAS ----------
            if len(bias_buf) < BIAS_SAMPLES:
                bias_buf.append((acc, gyro))
                if len(bias_buf) % 50 == 0:
                    print(f"Bias samples: {len(bias_buf)}/{BIAS_SAMPLES}")
                if len(bias_buf) == BIAS_SAMPLES:
                    acc_mean = np.mean([b[0] for b in bias_buf], axis=0)
                    gyro_mean = np.mean([b[1] for b in bias_buf], axis=0)
                    acc_bias[:] = acc_mean - np.array([0, 0, G])
                    gyro_bias[:] = gyro_mean
                    print("Bias locked.")
                    print("ACC bias:", acc_bias)
                    print("GYRO bias:", gyro_bias)
                continue

            # ---------- REMOVE BIAS ----------
            acc_corr = acc - acc_bias
            gyro_corr = gyro - gyro_bias

            # ---------- ORIENTATION ----------
            gyro_rad = np.deg2rad(gyro_corr)
            acc_norm_mag = np.linalg.norm(acc_corr)
            if acc_norm_mag > 1e-3:
                acc_norm = acc_corr / acc_norm_mag
            else:
                acc_norm = acc_corr
            quat = madgwick.updateIMU(quat, gyro_rad, acc_norm)
            orientation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])


            # ---------- ACC → WORLD ----------
            acc_world = orientation.apply(acc_corr)
            acc_world -= np.array([0.0, 0.0, G])
            # ---------- Stationary Detection----------
            stationary = ( abs(acc_norm_mag - G) < ZUPT_ACC and np.linalg.norm(gyro_corr) < ZUPT_GYRO)
               
            # ---------- Velocity ----------
            vel += acc_world * dt
            
            if stationary:
                vel[:] = 0.0
            # ---------- Position ----------
            pos += vel * dt
            #---------- LOGGING ----------
            yaw = orientation.as_euler("xyz", degrees=True)[2]
            writer.writerow([now, pos[0], pos[1], pos[2], yaw])

            if now - last_print_time >= PRINT_INTERVAL:
                last_print_time = now
                print(
                    f"dt={dt:.4f}s | "
                    f"POS: x={pos[0]:.3f} y={pos[1]:.3f} | "
                    f"VEL: {np.linalg.norm(vel):.3f} m/s | "
                    f"YAW={yaw:.1f}°",
                    flush=True
                )

    print("Imu data saved :", CSV_FILE)

if __name__ == "__main__":
    main()