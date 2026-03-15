import json
from datetime import datetime
import math

import matplotlib.pyplot as plt
import numpy as np

# =========================
# Config
# =========================
FILE_PATH = "imu_data.json" # based on output file name 

# Moving average window sizes (in samples)
ACC_WINDOW = 5
GYRO_WINDOW = 5

# Complementary filter parameter for orientation
# alpha close to 1 = trust gyro more, alpha close to 0 = trust accel more
ORIENT_ALPHA = 0.98


# =========================
# Helper functions
# =========================
def moving_average(data, window):
    """Simple centered moving average."""
    if window <= 1:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def parse_timestamp(ts_str):
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(ts_str)


# =========================
# Load JSON data
# =========================
with open(FILE_PATH, "r") as f:
    samples = json.load(f)

timestamps = []
t_sec = []  # seconds since start

acc_x = []
acc_y = []
acc_z = []

gyro_x_dps = []
gyro_y_dps = []
gyro_z_dps = []

tow = []

for entry in samples:
    # Timestamp
    ts = parse_timestamp(entry["timestamp"])
    timestamps.append(ts)

    # accel
    la = entry["linear_acceleration"]
    acc_x.append(la["x_mps2"])
    acc_y.append(la["y_mps2"])
    acc_z.append(la["z_mps2"])

    # gyro (deg/s)
    av = entry["angular_velocity"]
    gyro_x_dps.append(av["x_dps"])
    gyro_y_dps.append(av["y_dps"])
    gyro_z_dps.append(av["z_dps"])

    # GPS time-of-week (optional, but might be useful)
    tow.append(entry["tow"])

# Convert timestamps to seconds since start
t0 = timestamps[0]
for ts in timestamps:
    dt = (ts - t0).total_seconds()
    t_sec.append(dt)

t_sec = np.array(t_sec)
acc_x = np.array(acc_x)
acc_y = np.array(acc_y)
acc_z = np.array(acc_z)
gyro_x_dps = np.array(gyro_x_dps)
gyro_y_dps = np.array(gyro_y_dps)
gyro_z_dps = np.array(gyro_z_dps)

# =========================
# Convert gyro to rad/s
# =========================
deg2rad = math.pi / 180.0
gyro_x = gyro_x_dps * deg2rad
gyro_y = gyro_y_dps * deg2rad
gyro_z = gyro_z_dps * deg2rad

# =========================
# Moving-average filtering
# =========================
acc_x_f = moving_average(acc_x, ACC_WINDOW)
acc_y_f = moving_average(acc_y, ACC_WINDOW)
acc_z_f = moving_average(acc_z, ACC_WINDOW)

gyro_x_f = moving_average(gyro_x, GYRO_WINDOW)
gyro_y_f = moving_average(gyro_y, GYRO_WINDOW)
gyro_z_f = moving_average(gyro_z, GYRO_WINDOW)

# =========================
# Total acceleration magnitude
# =========================
acc_mag = np.sqrt(acc_x_f**2 + acc_y_f**2 + acc_z_f**2)

# =========================
# Orientation estimation (roll & pitch)
#
# roll  ~ rotation around X axis
# pitch ~ rotation around Y axis
#
# We use a simple complementary filter:
#   roll  = alpha*(roll + gx*dt)  + (1-alpha)*roll_acc
#   pitch = alpha*(pitch + gy*dt) + (1-alpha)*pitch_acc
#
# No yaw estimation here (requires magnetometer or external reference).
# =========================

roll  = np.zeros_like(t_sec)  # radians
pitch = np.zeros_like(t_sec)  # radians

if len(t_sec) > 1:
    for i in range(1, len(t_sec)):
        dt = t_sec[i] - t_sec[i - 1]
        if dt <= 0:
            dt = 1e-3  # fallback small dt if timestamps are equal / out-of-order

        # Integrate gyro to get orientation change (gyro_x_f/gyro_y_f already rad/s)
        roll_gyro  = roll[i - 1]  + gyro_x_f[i] * dt
        pitch_gyro = pitch[i - 1] + gyro_y_f[i] * dt

        # Compute roll & pitch from accelerometer (tilt w.r.t. gravity)
        # Convention:
        #   roll_acc  = atan2(ay, az)
        #   pitch_acc = atan2(-ax, sqrt(ay^2 + az^2))
        ax = acc_x_f[i]
        ay = acc_y_f[i]
        az = acc_z_f[i]

        # avoid divide by zero
        denom = math.sqrt(ay*ay + az*az) if (ay != 0.0 or az != 0.0) else 1e-6

        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, denom)

        # Complementary filter
        roll[i]  = ORIENT_ALPHA * roll_gyro  + (1.0 - ORIENT_ALPHA) * roll_acc
        pitch[i] = ORIENT_ALPHA * pitch_gyro + (1.0 - ORIENT_ALPHA) * pitch_acc

# Convert roll/pitch to degrees for plotting
roll_deg  = roll * 180.0 / math.pi
pitch_deg = pitch * 180.0 / math.pi

# =========================
# PLOTS
# =========================

# ---- 1) Linear acceleration ----
plt.figure(figsize=(12, 6))
plt.plot(t_sec, acc_x_f, label="acc_x (m/s²)")
plt.plot(t_sec, acc_y_f, label="acc_y (m/s²)")
plt.plot(t_sec, acc_z_f, label="acc_z (m/s²)")
plt.title("IMU Linear Acceleration ")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True)

# ---- 2) Angular velocity (rad/s) ----
plt.figure(figsize=(12, 6))
plt.plot(t_sec, gyro_x_f, label="gyro_x (rad/s)")
plt.plot(t_sec, gyro_y_f, label="gyro_y (rad/s)")
plt.plot(t_sec, gyro_z_f, label="gyro_z (rad/s)")
plt.title("IMU Angular Velocity -> rad/s)")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.grid(True)

# ---- 3) Total acceleration magnitude ----
plt.figure(figsize=(12, 4))
plt.plot(t_sec, acc_mag, label="|acc| (m/s²)")
plt.title("Total Acceleration Magnitude")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True)

# ---- 4) Orientation (roll & pitch) ----
plt.figure(figsize=(12, 6))
plt.plot(t_sec, roll_deg, label="roll (deg)")
plt.plot(t_sec, pitch_deg, label="pitch (deg)")
plt.title("Estimated Orientation (roll/pitch from IMU)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.legend()
plt.grid(True)

plt.show()
