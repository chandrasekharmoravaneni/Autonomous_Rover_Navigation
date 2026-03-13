
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Select experiment ----------------
trajectory = "square"   # square | circle | straight
fix_mode = "SBAS"       # SPP | SBAS

# ---------------- Load data ----------------
df = pd.read_csv(f"realtime_gnss_data.csv")

# ---------------- Convert lat/lon → local meters ----------------
lat0 = np.deg2rad(df.lat_deg.mean())
lon0 = np.deg2rad(df.lon_deg.mean())
R = 6378137.0

x = R * (np.deg2rad(df.lon_deg) - lon0) * np.cos(lat0)
y = R * (np.deg2rad(df.lat_deg) - lat0)

# ---------------- Plot track ----------------
plt.figure()
plt.plot(x, y, ".", markersize=3)
plt.axis("equal")
plt.grid(True)
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title(f"{fix_mode} Trajectory ({trajectory.capitalize()})")
plt.show()
