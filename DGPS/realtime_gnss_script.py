"""
GNSS Position Logger
Logs position, accuracy, satellites, and GNSS fix type (SPP/DGPS/RTK).
Ctrl+C to stop.
"""
from sbp.client.drivers.network_drivers import TCPDriver
from sbp.client import Framer
from sbp.navigation import MsgPosLLH
import csv
import time

ROVER_IP = "195.37.48.233" # Piski rover ip address
ROVER_PORT = 55555         # Piski rover port

driver = TCPDriver(IP, PORT)
framer = Framer(driver.read, driver.write)

csv_file = open("realtime_gnss_data.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "timestamp",
    "lat_deg",
    "lon_deg",
    "height_m",
    "h_accuracy_m",
    "n_sats",
    "fix"
])

print("Logging position + height..... Ctrl+C to stop\n")

try:
    for msg, meta in framer:

        if isinstance(msg, MsgPosLLH):
            fix = msg.flags & 0x7

            writer.writerow([
                time.time(),
                msg.lat,
                msg.lon,
                msg.height,        
                msg.h_accuracy,
                msg.n_sats,
                fix
            ])

            print(
                f"Fix={fix} | "
                f"Height={msg.height:7.2f} m | "
                f"Hacc={msg.h_accuracy:5.2f} m | "
                f"Lat={msg.lat:.8f} Lon={msg.lon:.8f}"
            )

except KeyboardInterrupt:
    print("\nStopped logging")

finally:
    csv_file.close()
    print("CSV saved: realtime_gnss_data.csv")

