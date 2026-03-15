"""
tcp_dgps_connection_test.py

This script checks the TCP connection to the DGPS/INS device to verify that the
socket is reachable and responding. It attempts to establish a TCP connection,
reports connection status, and helps diagnose connectivity issues before running
real-time GNSS or IMU data streaming scripts.
"""

import socket

# Port and IP for the DGPS device
IP = "195.37.48.233"   
PORT = 55555          

try:
    print(f"Connecting to DGPS {IP}:{PORT} ...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)   # 5 seconds timeout
    sock.connect((IP, PORT))

    print(" Connected! Reading incoming DGPS data...\n")

    while True:
        data = sock.recv(4096)
        if not data:
            print(" No more data received.")
            break

        print(" Received:", data.hex())   # hex format (SBP raw)
        # print(data)                      # uncomment for ASCII/NMEA

except socket.timeout:
    print("Connection timeout. Device not reachable.")
except ConnectionRefusedError:
    print("Connection refused. DGPS port not open.")
except OSError as e:
    print(" OS Error:", e)
finally:
    try:
        sock.close()
    except:
        pass

print("Finished.")
