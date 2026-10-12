"""Example UDP client for `python -m src.sim live`: sends one excitation in [0, 1] per muscle at 100 Hz.

Alternates finger flexors and extensors so the hand opens and closes. Muscle order is read from the model.
"""
import math
import socket
import struct
import time

from .model import load

HOST, PORT = "127.0.0.1", 5006
m, _ = load()
names = [m.actuator(i).name for i in range(m.nu)]
FLEXORS = [i for i, n in enumerate(names) if n.startswith(("FDS", "FDP"))]
EXTENSORS = [i for i, n in enumerate(names) if n.startswith(("EDC", "EDM", "EIP"))]

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
u = [0.0] * m.nu
try:
    while True:
        s = 0.5 + 0.5 * math.sin(time.time() * 2)
        for i in FLEXORS:
            u[i] = 0.4 * s
        for i in EXTENSORS:
            u[i] = 0.4 * (1 - s)
        sock.sendto(struct.pack(f"{m.nu}f", *u), (HOST, PORT))
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Stopped controller client.")
finally:
    sock.close()
