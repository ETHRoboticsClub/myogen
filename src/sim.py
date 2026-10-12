"""Forearm web viewer (mjviser, http://localhost:8080).

    python -m src.sim live                         # muscle excitations from UDP (nu floats in [0, 1])
    python -m src.sim demo                         # live computed-muscle-control of random natural motion
    python -m src.sim replay data/pilot/shard_000.h5 [trial]   # play back a generated trial
"""
import socket
import struct
import sys

import h5py
import mujoco
import numpy as np
from mjviser import Viewer

from .model import Anatomy, load
from .motion import JointMap, intent_trajectory
from .tracker import MuscleTracker, TrackerConfig

PORT = 5006   # e3m5_hand uses 5005


def live(m, d):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.setblocking(False)
    fmt = f"{m.nu}f"
    size = struct.calcsize(fmt)

    def step(m, d):
        packet = None
        try:
            while True:                      # drain: always act on the newest packet
                packet = sock.recv(size)
        except BlockingIOError:
            pass
        if packet is not None and len(packet) == size:
            d.ctrl[:] = np.clip(struct.unpack(fmt, packet), 0.0, 1.0)
        mujoco.mj_step(m, d)

    print(f"listening for {m.nu} muscle excitations on udp://0.0.0.0:{PORT}")
    print("order:", " ".join(m.actuator(i).name for i in range(m.nu)))
    return step


def demo(m, d):
    anat = Anatomy.from_model(m, d)
    rng = np.random.default_rng()
    fs = 100
    every = int(round(1 / (fs * m.opt.timestep)))
    jmap = JointMap.sample(rng, *anat.hand_limits())
    tracker = MuscleTracker(m, d, TrackerConfig(ctrl_every=every))
    state = {"k": 0, "sub": 0, "q": None}

    def new_segment():
        q = anat.full_q(jmap(intent_trajectory(rng, 20.0, fs)), elbow=1.2)
        state.update(q=q, qd=np.gradient(q, 1 / fs, axis=0), k=0)
        state["qdd"] = np.gradient(state["qd"], 1 / fs, axis=0)

    new_segment()
    d.qpos[:] = state["q"][0]
    mujoco.mj_forward(m, d)

    def step(m, d):
        if state["sub"] == 0:
            k = state["k"]
            if k >= len(state["q"]):
                new_segment()
                k = 0
            u, _, _ = tracker.control(state["q"][k], state["qd"][k], state["qdd"][k], 0.0, np.zeros(m.nv))
            d.ctrl[:] = u
            state["k"] = k + 1
        state["sub"] = (state["sub"] + 1) % every
        mujoco.mj_step(m, d)

    return step


def replay(m, d, path, trial=None):
    with h5py.File(path, "r") as f:
        trial = trial or sorted(f.keys())[0]
        g = f[trial]
        q, act = g["labels/q"][:], g["labels/act"][:]
        fs = float(g.attrs["fs_label"])
    print(f"replaying {trial}: {len(q) / fs:.1f}s")

    def step(m, d):
        d.time += m.opt.timestep
        k = int(d.time * fs) % len(q)
        d.qpos[:] = q[k]
        d.act[:] = act[k]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

    return step


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    m, d = load()
    if mode == "live":
        step = live(m, d)
    elif mode == "demo":
        step = demo(m, d)
    elif mode == "replay":
        step = replay(m, d, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        sys.exit(__doc__)
    Viewer(m, d, step_fn=step).run()


if __name__ == "__main__":
    main()
