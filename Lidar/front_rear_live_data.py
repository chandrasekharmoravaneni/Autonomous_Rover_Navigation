#!/usr/bin/env python3

import socket
import struct
import math
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import atexit

# ================================================================
# CONFIGURATION
# ================================================================
MAX_RANGE = 25.0
MASK_RADIUS = 0.0
STX = b"\x02"
ETX = b"\x03"

LIDARS = [
    {
        "ip": "195.37.48.222",
        "port": 2111,
        "name": "Front LiDAR",
        "yaw_deg": 0.0,
        "offset": (0.0, 0.3),
        "color": (0, 200, 255)
    },
    {
        "ip": "195.37.48.223",
        "port": 2111,
        "name": "Rear LiDAR",
        "yaw_deg": 180.0,
        "offset": (0.0, -0.3),
        "color": (255, 120, 0)
    }
]

# ================================================================
# SESSION STATS
# ================================================================
session_start = time.time()

lidar_stats = {
    "Front LiDAR": {"frames": 0, "connected": False},
    "Rear LiDAR":  {"frames": 0, "connected": False},
}

# ================================================================
# TRANSFORM & PARSE
# ================================================================
def parse_lmd_scandata(block):
    parts = block.strip().split()
    if b"LMDscandata" not in parts or b"DIST1" not in parts:
        return None

    try:
        i = parts.index(b"DIST1")
        scale_hex = parts[i + 1].decode()
        scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
        start = int(parts[i + 3], 16) / 10000.0
        step  = int(parts[i + 4], 16) / 10000.0
        count = int(parts[i + 5], 16)

        ranges = []
        for h in parts[i + 6 : i + 6 + count]:
            val = int(h, 16)
            ranges.append((val * scale) / 1000.0 if val > 0 else 0)

        ranges = np.array(ranges)
        angles_deg = start + np.arange(count) * step

        valid = (ranges > 0.01) & (ranges <= MAX_RANGE + 1.0)
        r = ranges[valid]
        a = np.radians(angles_deg[valid])

        lx = r * np.cos(a)
        ly = r * np.sin(a)

        return lx, ly

    except Exception as e:
        print(f"Parse error: {e}")
        return None


def apply_transform(lx, ly, yaw_deg, offset):
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    tx, ty = offset

    gx = (c * lx - s * ly) + tx
    gy = (s * lx + c * ly) + ty
    return gx, gy

# ================================================================
# LIDAR CLIENT
# ================================================================
class LidarClient(QtCore.QObject):
    def __init__(self, cfg, scatter):
        super().__init__()
        self.cfg = cfg
        self.scatter = scatter
        self.sock = None
        self.buffer = b""

        self.init_connection()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(20)

    def init_connection(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.0)
            self.sock.connect((self.cfg["ip"], self.cfg["port"]))

            for cmd in ("sMN LMCstartmeas", "sMN Run", "sEN LMDscandata 1"):
                self.sock.sendall(STX + cmd.encode() + ETX)
                time.sleep(0.05)

            self.sock.settimeout(0.001)
            lidar_stats[self.cfg["name"]]["connected"] = True
            print(f"[ONLINE] {self.cfg['name']} active.")

        except Exception as e:
            print(f"[OFFLINE] {self.cfg['name']} failed: {e}")

    def poll(self):
        if not self.sock:
            return

        try:
            while True:
                data = self.sock.recv(65535)
                if not data:
                    break
                self.buffer += data
                if len(self.buffer) > 40000:
                    self.buffer = self.buffer[-8000:]
        except:
            pass

        while STX in self.buffer and ETX in self.buffer:
            s = self.buffer.find(STX)
            e = self.buffer.find(ETX, s)
            if e == -1:
                break

            telegram = self.buffer[s + 1:e]
            self.buffer = self.buffer[e + 1:]

            parsed = parse_lmd_scandata(telegram)
            if parsed is None:
                continue

            lx, ly = parsed
            gx, gy = apply_transform(
                lx, ly,
                self.cfg["yaw_deg"],
                self.cfg["offset"]
            )

            dist = np.sqrt(gx**2 + gy**2)
            mask = (dist > MASK_RADIUS) & (dist <= MAX_RANGE)

            lidar_stats[self.cfg["name"]]["frames"] += 1
            self.scatter.setData(gx[mask], gy[mask])

# ================================================================
# FINAL SUMMARY
# ================================================================
def print_summary():
    duration = time.time() - session_start

    print("\n" + "=" * 40)
    print("        FINAL SESSION SUMMARY")
    print("=" * 40)

    for cfg in LIDARS:
        name = cfg["name"]
        ip = cfg["ip"]
        frames = lidar_stats[name]["frames"]
        connected = lidar_stats[name]["connected"]

        hz = frames / duration if duration > 0 else 0
        fpm = hz * 60

        print(f"Device: {name} [{ip}]")
        print(f"Status: {'CONNECTED' if connected else 'DISCONNECTED'}")
        print(f"Frames: {frames}")
        print(f"Avg Rate: {hz:.2f} Hz ({int(fpm)} Frames Per Minute)")
        print("-" * 40)

    print("Process Terminated.")

atexit.register(print_summary)

# ================================================================
# MAIN UI
# ================================================================
def main():
    app = QtWidgets.QApplication([])

    win = pg.GraphicsLayoutWidget(show=True, title="SICK DUAL TiM781 - 360°  Alignment")
    win.resize(900, 900)
    win.setBackground('k')

    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.showGrid(x=False, y=False)
    plot.setXRange(-27, 27)
    plot.setYRange(-27, 27)

    xAxis = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=1, alpha=150))
    yAxis = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=1, alpha=150))
    plot.addItem(xAxis)
    plot.addItem(yAxis)

    for r in range(5, 26, 5):
        circle = QtWidgets.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
        circle.setPen(pg.mkPen((70, 70, 70), width=1, style=QtCore.Qt.DotLine))
        plot.addItem(circle)
        text = pg.TextItem(text=f"{r}m", color=(100, 100, 100), anchor=(0.5, 1))
        text.setPos(0, r)
        plot.addItem(text)

    scatters = {}
    for cfg in LIDARS:
        scatters[cfg["name"]] = plot.plot(
            pen=None, symbol='o', symbolSize=2, symbolBrush=cfg["color"]
        )
        plot.plot(
            [cfg["offset"][0]], [cfg["offset"][1]],
            pen=None, symbol='+', symbolSize=15,
            symbolBrush=cfg["color"]
        )

    plot.plot([0], [0], pen=None, symbol='s', symbolSize=8, symbolBrush='r')

    clients = [LidarClient(cfg, scatters[cfg["name"]]) for cfg in LIDARS]

    app.exec_()

if __name__ == "__main__":
    main()