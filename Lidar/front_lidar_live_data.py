#!/usr/bin/env python3

import socket
import struct
import math
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# ================================================================
# CONFIGURATION
# ================================================================
MAX_RANGE = 30.0        # meters
STX = "\x02"
ETX = "\x03"
# Front LiDAR configured; rear LiDAR is tested individually using IP address 195.37.48.223

LIDARS_CONFIG = [
    {
        "ip": "195.37.48.222",
        "port": 2111,
        "color": (0, 200, 255),
        "dx": 0.0,
        "dy": 0.0,
        "yaw_deg": 0.0,
        "name": "Front_LiDAR"
    }
]

# ================================================================
# SICK TELEGRAM PARSING (CARTESIAN)
# ================================================================

def parse_lmd_scandata(block):
    parts = block.strip().split()
    if b"LMDscandata" not in parts or b"DIST1" not in parts:
        return None

    try:
        i = parts.index(b"DIST1")

        scale = struct.unpack('>f', bytes.fromhex(parts[i + 1].decode()))[0]
        start = int(parts[i + 3], 16) / 10000.0
        step  = int(parts[i + 4], 16) / 10000.0
        count = int(parts[i + 5], 16)

        vals = parts[i + 6:i + 6 + count]

        xs, ys = [], []
        for k, h in enumerate(vals):
            raw = int(h, 16)
            if 0 < raw < 0xFFFF:
                r = (raw * scale) / 1000.0
                angle = math.radians(start + k * step)

                # ---- STANDARD POLAR → CARTESIAN ----
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                xs.append(x)
                ys.append(y)

        return xs, ys

    except Exception as e:
        print("Parse error:", e)
        return None


def transform_points(xs, ys, dx, dy, yaw_deg):
    if yaw_deg == 0.0:
        return [x + dx for x in xs], [y + dy for y in ys]

    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return (
        [c * x - s * y + dx for x, y in zip(xs, ys)],
        [s * x + c * y + dy for x, y in zip(xs, ys)]
    )

# ================================================================
# LIDAR CLIENT
# ================================================================

class LidarClient(QtCore.QObject):
    def __init__(self, config, plot_item):
        super().__init__()
        self.cfg = config
        self.plot_item = plot_item
        self.sock = None
        self.buffer = b""
        self.connected = False

        self.init_connection()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(20)

    def init_connection(self):
        try:
            print(f"Connecting to {self.cfg['name']}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.cfg["ip"], self.cfg["port"]))

            for cmd in ("sMN LMCstartmeas", "sMN Run", "sEN LMDscandata 1"):
                self.sock.sendall(f"{STX}{cmd}{ETX}".encode("ascii"))
                time.sleep(0.15)

            self.sock.settimeout(0.001)
            self.connected = True
            print(f"[OK] {self.cfg['name']} streaming")

        except Exception as e:
            print(f"[ERROR] {self.cfg['name']} -> {e}")

    def poll(self):
        if not self.connected:
            return

        try:
            while True:
                data = self.sock.recv(65535)
                if not data:
                    break
                self.buffer += data
        except:
            pass

        while STX.encode() in self.buffer and ETX.encode() in self.buffer:
            s = self.buffer.find(STX.encode())
            e = self.buffer.find(ETX.encode(), s + 1)
            if s < 0 or e < 0:
                break

            telegram = self.buffer[s + 1:e]
            self.buffer = self.buffer[e + 1:]

            parsed = parse_lmd_scandata(telegram)
            if parsed:
                tx, ty = transform_points(
                    parsed[0], parsed[1],
                    self.cfg["dx"], self.cfg["dy"], self.cfg["yaw_deg"]
                )
                self.plot_item.setData(tx, ty)

# ================================================================
# STYLE CIRCULAR FAN (CARTESIAN)
# ================================================================

def add_circular_fan(plot):
    FOV_MIN_DEG = -45
    FOV_MAX_DEG = 225
    R_MIN = 0.5
    R_MAX = 25.0

    angles = np.linspace(
        math.radians(FOV_MIN_DEG),
        math.radians(FOV_MAX_DEG),
        600
    )

    # Outer arc
    x_outer = [R_MAX * math.cos(a) for a in angles]
    y_outer = [R_MAX * math.sin(a) for a in angles]

    # Inner arc (reverse)
    x_inner = [R_MIN * math.cos(a) for a in reversed(angles)]
    y_inner = [R_MIN * math.sin(a) for a in reversed(angles)]

    # Closed polygon
    x_fan = x_outer + x_inner + [x_outer[0]]
    y_fan = y_outer + y_inner + [y_outer[0]]

    plot.addItem(pg.PlotDataItem(
        x_fan,
        y_fan,
        pen=pg.mkPen((120, 0, 0), width=1),
        brush=pg.mkBrush(200, 80, 80, 150)
    ))

    # Range rings
    for r in range(5, int(R_MAX) + 1, 5):
        x = [r * math.cos(a) for a in angles]
        y = [r * math.sin(a) for a in angles]
        plot.plot(
            x, y,
            pen=pg.mkPen((150, 150, 150), width=1, style=QtCore.Qt.DotLine)
        )

    # FoV boundary lines
    for deg in (FOV_MIN_DEG, FOV_MAX_DEG):
        a = math.radians(deg)
        plot.plot(
            [0, R_MAX * math.cos(a)],
            [0, R_MAX * math.sin(a)],
            pen=pg.mkPen((80, 80, 80), width=2)
        )

    # Origin
    plot.plot([0], [0], pen=None, symbol='o', symbolSize=8, symbolBrush='k')

# ================================================================
# MAIN APPLICATION
# ================================================================

def main():
    app = QtWidgets.QApplication([])

    win = pg.GraphicsLayoutWidget(
        show=True,
        title="SICK TiM781 – Circular Fan View (Cartesian X,Y)"
    )
    win.resize(900, 900)

    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.setXRange(-MAX_RANGE, MAX_RANGE)
    plot.setYRange(-MAX_RANGE, MAX_RANGE)

    plot.setLabel('bottom', "X (m)")
    plot.setLabel('left', "Y (m)")
    plot.showGrid(x=True, y=True, alpha=0.5)

    add_circular_fan(plot)

    clients = []
    for cfg in LIDARS_CONFIG:
        scatter = plot.plot(
            pen=None,
            symbol='o',
            symbolSize=2,
            symbolBrush=cfg["color"]
        )
        clients.append(LidarClient(cfg, scatter))

    def cleanup():
        for c in clients:
            if c.sock:
                try:
                    c.sock.sendall(f"{STX}sEN LMDscandata 0{ETX}".encode())
                    c.sock.close()
                except:
                    pass
        print("System offline.")

    app.aboutToQuit.connect(cleanup)
    app.exec_()

if __name__ == "__main__":
    main()
