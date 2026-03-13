"""
Realtime_data_dgps_simulation.py

This script connects to the DGPS/INS receiver over TCP and continuously reads
real-time GNSS and IMU data. The incoming SBP messages are parsed and stored
in JSON format for logging and offline analysis. The script also generates live
matplotlib visualizations to monitor GNSS trajectories and IMU sensor behavior
during real-time operation.
"""


import json
import matplotlib.pyplot as plt

# Load GNSS JSON
with open("gnss_data.json", "r") as f:
    gnss = json.load(f)

# Load IMU JSON
with open("imu_data.json", "r") as f:
    imu = json.load(f)

# ------------ GNSS ------------
gnss_lat = []
gnss_lon = []
gnss_vn = []
gnss_ve = []
gnss_tow = []

for item in gnss:
    if "lat" in item:
        gnss_lat.append(item["lat"])
        gnss_lon.append(item["lon"])
        gnss_tow.append(item["tow"])
    if "vel_n_mps" in item:
        gnss_vn.append(item["vel_n_mps"])
        gnss_ve.append(item["vel_e_mps"])

# ------------  IMU ------------
imu_ax = []
imu_ay = []
imu_az = []
imu_gx = []
imu_gy = []
imu_gz = []
imu_tow = []

for item in imu:
    if "acc_x_mps2" in item:
        imu_ax.append(item["acc_x_mps2"])
        imu_ay.append(item["acc_y_mps2"])
        imu_az.append(item["acc_z_mps2"])
        imu_gx.append(item["gyr_x_dps"])
        imu_gy.append(item["gyr_y_dps"])
        imu_gz.append(item["gyr_z_dps"])
        imu_tow.append(item["tow"])

# ------------ PLOT GNSS LAT/LON ------------
plt.figure(figsize=(6,6))
plt.plot(gnss_lon, gnss_lat, '.-')
plt.title("GNSS Track (Lat/Lon)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid()
plt.show()

# ------------ PLOT IMU ACCEL ------------
plt.figure(figsize=(10,4))
plt.plot(imu_ax, label="AX")
plt.plot(imu_ay, label="AY")
plt.plot(imu_az, label="AZ")
plt.title("IMU Accelerometer (m/s²)")
plt.legend()
plt.grid()
plt.show()

# ------------ PLOT IMU GYRO ------------
plt.figure(figsize=(10,4))
plt.plot(imu_gx, label="GX")
plt.plot(imu_gy, label="GY")
plt.plot(imu_gz, label="GZ")
plt.title("IMU Gyroscope (deg/s)")
plt.legend()
plt.grid()
plt.show()

# ------------ PLOT GNSS VELOCITY ------------
plt.figure(figsize=(10,4))
plt.plot(gnss_vn, label="V North")
plt.plot(gnss_ve, label="V East")
plt.title("GNSS Velocities (m/s)")
plt.legend()
plt.grid()
plt.show()

print("GNSS samples:", len(gnss))
print("IMU samples:", len(imu))
