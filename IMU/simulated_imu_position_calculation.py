#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R


# ================= CONFIG =================
RATE = 100
DT = 1.0 / RATE
DURATION = 6.0
G = 9.80665

ACC_RANGE_G = 2
GYRO_RANGE_DPS = 2000

ACC_SCALE = ACC_RANGE_G * G / 32768.0
GYRO_SCALE = GYRO_RANGE_DPS / 32768.0

samples = int(RATE * DURATION)

# ---- Disturbances ----
ACC_BIAS = np.array([0.02, -0.01, 0.0])   # m/s^2
GYRO_BIAS = np.array([0.0, 0.0, 0.5])     # deg/s bias on Z

ACC_NOISE_STD = 0.02
GYRO_NOISE_STD = 0.1

# ================= STATE =================
pos_est = np.zeros(2)
vel_est = np.zeros(2)


pos_gt = np.zeros(2)
vel_gt = np.zeros(2)
yaw_gt = 0.0

# ---- Madgwick Filter ----
madgwick = Madgwick(beta=0.1)
quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

# Logging
time_arr = []
pos_est_arr = []
pos_gt_arr = []
vel_est_arr = []
error_arr = []
pos_gt_xy = []
pos_est_xy = []

# ================= SIMULATION =================
for i in range(samples):

    t = i * DT

    # ---- Motion profile (3m in 6s) ----
    if t < 1.0:
        ax_body = 0.6
    elif t < 5.0:
        ax_body = 0.0
    else:
        ax_body = -0.6

    # ---- Yaw rotation during motion ----
    gz_true = 15 if 1.5 < t < 4.5 else 0.0  # deg/s
    yaw_gt += np.deg2rad(gz_true) * DT

    # ---- Ground Truth Transform ----
    R_gt = np.array([
        [np.cos(yaw_gt), -np.sin(yaw_gt)],
        [np.sin(yaw_gt),  np.cos(yaw_gt)]
    ])

    acc_world_gt = R_gt @ np.array([ax_body, 0.0])
    vel_gt += acc_world_gt * DT
    pos_gt += vel_gt * DT

    # ---- Simulated IMU (BMI160 style) ----
    acc_body = np.array([ax_body, 0.0, G])
    gyro_body = np.array([0.0, 0.0, gz_true])

    # Add bias + noise
    acc_body += ACC_BIAS + np.random.normal(0, ACC_NOISE_STD, 3)
    gyro_body += GYRO_BIAS + np.random.normal(0, GYRO_NOISE_STD, 3)

    # Convert to raw and back (quantization)
    raw_acc = (acc_body / ACC_SCALE).astype(int)
    raw_gyro = (gyro_body / GYRO_SCALE).astype(int)

    acc = raw_acc * ACC_SCALE
    gyro = raw_gyro * GYRO_SCALE

    # ---- Estimation ----
    gyro_rad = np.deg2rad(gyro)
    acc_norm = acc / np.linalg.norm(acc)

    quat = madgwick.updateIMU(quat, gyr=gyro_rad, acc=acc_norm)
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    yaw_est = rot.as_euler('xyz')[2]
    R_est = np.array([
        [np.cos(yaw_est), -np.sin(yaw_est)],
        [np.sin(yaw_est),  np.cos(yaw_est)]
    ])

    acc_world_est = R_est @ acc[:2]

    vel_est += acc_world_est * DT
    pos_est += vel_est * DT

    # ---- Logging ----
    time_arr.append(t)
    pos_est_arr.append(np.linalg.norm(pos_est))
    pos_gt_arr.append(np.linalg.norm(pos_gt))
    vel_est_arr.append(np.linalg.norm(vel_est))
    error_arr.append(np.linalg.norm(pos_est - pos_gt))
    pos_gt_xy.append(pos_gt.copy())
    pos_est_xy.append(pos_est.copy())

# Convert logs to arrays
pos_gt_xy = np.array(pos_gt_xy)
pos_est_xy = np.array(pos_est_xy)

# ================= RESULTS =================
bias_mag = np.linalg.norm(ACC_BIAS[:2])
predicted_drift = 0.5 * bias_mag * DURATION**2

print("\n==== STRESS TEST RESULT ====")
print("Final GT Position:", np.linalg.norm(pos_gt))
print("Final Estimated Position:", np.linalg.norm(pos_est))
print("Final Velocity:", np.linalg.norm(vel_est))
print("Final Position Error:", np.linalg.norm(pos_est - pos_gt))
print("Predicted Drift from Bias (theory):", predicted_drift)
print("============================\n")

# ================= PLOTS =================

# Position magnitude
plt.figure()
plt.plot(time_arr, pos_gt_arr, label="Ground Truth")
plt.plot(time_arr, pos_est_arr, "--", label="Estimated")
plt.title("Position vs Time (Bias + Noise + Rotation)")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()
plt.grid()
plt.show()

# Position error growth
plt.figure()
plt.plot(time_arr, error_arr)
plt.title("Position Error Growth")
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.grid()
plt.show()

# Velocity magnitude
plt.figure()
plt.plot(time_arr, vel_est_arr)
plt.title("Velocity Magnitude")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid()
plt.show()

# 2D Trajectory
plt.figure()
plt.plot(pos_gt_xy[:,0], pos_gt_xy[:,1], label="Ground Truth")
plt.plot(pos_est_xy[:,0], pos_est_xy[:,1], '--', label="Estimated")
plt.title("2D Trajectory")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis('equal')
plt.legend()
plt.grid()
plt.show()
