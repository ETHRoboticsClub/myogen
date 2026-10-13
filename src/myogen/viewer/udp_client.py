"""Example sender for `myogen view` in live mode: python -m myogen.viewer.udp_client

Sends one excitation in [0, 1] per muscle at 100 Hz, alternating finger flexors and extensors.
"""
import math
import socket
import struct
import time

UDP_PORT = 5006


def main(host="127.0.0.1", port=UDP_PORT):
    from ..body.model import load
    m, _ = load()
    names = [m.actuator(i).name for i in range(m.nu)]
    flexors = [i for i, n in enumerate(names) if n.startswith(("FDS", "FDP"))]
    extensors = [i for i, n in enumerate(names) if n.startswith(("EDC", "EDM", "EIP"))]
    u = [0.0] * m.nu
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        while True:
            s = 0.5 + 0.5 * math.sin(time.time() * 2)
            for i in flexors:
                u[i] = 0.4 * s
            for i in extensors:
                u[i] = 0.4 * (1 - s)
            sock.sendto(struct.pack(f"{m.nu}f", *u), (host, port))
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
