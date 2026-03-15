#!/usr/bin/env python3
"""
Minimal IMU logger (Python).

Outputs JSON array where each element has:
  - seq: integer sequence number
  - timestamp: ISO 8601 string (host time, UTC)
  - tow: GPS time-of-week from IMU (ms)
  - linear_acceleration: {x_mps2, y_mps2, z_mps2}
  - angular_velocity:   {x_dps,  y_dps,  z_dps}
"""

import json
import signal
from datetime import datetime

from sbp.client.drivers.network_drivers import TCPDriver
from sbp.client import Framer
from sbp.imu import MsgImuRaw, MsgImuAux

# --------------------------
# Config: IP / PORT
# --------------------------
IP   = "195.37.48.233"
PORT = 55555

# --------------------------
# Globals / state
# --------------------------
running = True

# Latest IMU configuration (from MsgImuAux)
acc_range_g = None        # +/- g
gyro_range_dps = None     # +/- deg/s
G_STANDARD = 9.80665      # m/s^2 per g

# Sequence number for IMU samples
seq = 0


def stop_handler(sig, frame):
    global running
    print("\nStopping logging (Ctrl + C)...\n")
    running = False


signal.signal(signal.SIGINT, stop_handler)


def decode_imu_conf(imu_conf):
    """
    Decode imu_conf bitfield from MsgImuAux to accel & gyro ranges.

    imu_conf bits:
      - bits 0..3  : accelerometer range code
      - bits 4..7  : gyroscope range code
    """
    # lower 4 bits: accel range
    acc_code = imu_conf & 0x0F
    # upper 4 bits: gyro range
    gyro_code = (imu_conf >> 4) & 0x0F

    # Accelerometer ranges [g]
    acc_ranges_g = {
        0: 2.0,    # +/- 2 g
        1: 4.0,    # +/- 4 g
        2: 8.0,    # +/- 8 g
        3: 16.0,   # +/- 16 g
        4: 6.0,    # +/- 6 g (if used)
    }

    # Gyroscope ranges [deg/s]
    gyro_ranges_dps = {
        0: 2000.0,  # +/- 2000 deg/s
        1: 1000.0,  # +/- 1000 deg/s
        2: 500.0,   # +/- 500 deg/s
        3: 250.0,   # +/- 250 deg/s
        4: 125.0,   # +/- 125 deg/s
        5: 300.0,   # +/- 300 deg/s
    }

    acc_range = acc_ranges_g.get(acc_code)
    gyro_range = gyro_ranges_dps.get(gyro_code)

    return acc_range, gyro_range


def main():
    global acc_range_g, gyro_range_dps, seq

    # Open output file
    out_file = open("imu_data.json", "w")
    out_file.write("[\n")
    first_record = True

    print(f"Connecting to DGPS/INS at {IP}:{PORT} ...")
    driver = TCPDriver(IP, PORT, timeout=5, reconnect=True)
    framer = Framer(driver.read, driver.write)
    print("Streaming SBP IMU messages for EKF...\n")

    for msg, meta in framer:

        if not running:
            break

        # Host timestamp (UTC)
        timestamp = datetime.utcnow().isoformat()

        # ---------- IMU_AUX ----------
        # Use this only to update scaling. We don't log it.
        if isinstance(msg, MsgImuAux):
            acc_range_g, gyro_range_dps = decode_imu_conf(msg.imu_conf)
            # Optional: uncomment for debug
            # print(f"IMU_AUX: acc_range_g={acc_range_g}, gyro_range_dps={gyro_range_dps}")
            continue

        # ---------- IMU_RAW ----------
        if isinstance(msg, MsgImuRaw):

            if acc_range_g is None or gyro_range_dps is None:
                # Haven't seen an IMU_AUX yet; skip until we know ranges
                # (or you could log raw only).
                # print("IMU_RAW received but no IMU_AUX yet; skipping.")
                continue

            # Raw readings from the IMU
            raw_ax = msg.acc_x
            raw_ay = msg.acc_y
            raw_az = msg.acc_z

            raw_gx = msg.gyr_x
            raw_gy = msg.gyr_y
            raw_gz = msg.gyr_z

            # Convert raw tics -> physical units
            # Accelerometer: g -> m/s^2
            acc_scale = acc_range_g * G_STANDARD / 32768.0
            ax = raw_ax * acc_scale
            ay = raw_ay * acc_scale
            az = raw_az * acc_scale

            # Gyroscope: deg/s
            gyro_scale = gyro_range_dps / 32768.0
            gx = raw_gx * gyro_scale
            gy = raw_gy * gyro_scale
            gz = raw_gz * gyro_scale

            seq += 1

            record = {
                "seq": seq,
                "timestamp": timestamp,
                "tow": msg.tow,  # GPS time-of-week in ms (useful for sync)
                "linear_acceleration": {
                    "x_mps2": ax,
                    "y_mps2": ay,
                    "z_mps2": az,
                },
                "angular_velocity": {
                    "x_dps": gx,
                    "y_dps": gy,
                    "z_dps": gz,
                },
            }

            if not first_record:
                out_file.write(",\n")
            first_record = False

            json.dump(record, out_file)
            # Optional: print to console for debugging
            print("IMU :", record, flush=True)

    # Clean exit
    out_file.write("\n]\n")
    out_file.close()
    print("\nJSON saved as imu_data.json\nDone.\n")


if __name__ == "__main__":
    main()
