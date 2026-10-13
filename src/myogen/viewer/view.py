"""`myogen view`: the forearm in the web viewer (mjviser, http://localhost:8080).

    replay   a generated trial with its recording setup and scrolling EMG (overlay.py)
    demo     live muscle control of random natural motion
    live     muscle excitations received over UDP (udp_client.py is an example sender)
"""
import logging
import socket
import struct

import h5py
import mujoco
import numpy as np
from mjviser import Viewer

from ..body.model import Anatomy, load
from ..body.tracker import MuscleTracker, TrackerConfig
from ..log import viser_server
from ..motion.latent import JointMap
from ..motion.synthetic import intent_trajectory
from .overlay import TrialOverlay
from .udp_client import UDP_PORT

log = logging.getLogger(__name__)


def live(m, d):
    """Apply the newest UDP packet of nu float32 excitations in [0, 1] at every step."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
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

    log.info(f"listening for {m.nu} muscle excitations on udp://0.0.0.0:{UDP_PORT}")
    log.info("order: " + " ".join(m.actuator(i).name for i in range(m.nu)))
    return step


def demo(m, d):
    """Track endless random natural motion with the muscle controller."""
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


def replay(m, d, server, path, trial=None, emg_window=4.0):
    """Loop a stored trial (first in the shard by default) with its recording-setup overlay."""
    with h5py.File(path, "r") as f:
        trial = trial or sorted(f.keys())[0]
        g = f[trial]
        q, act = g["labels/q"][:], g["labels/act"][:]
        fs = float(g.attrs["fs_label"])
        overlay = TrialOverlay.from_trial(server, m, d, g, q[0], emg_window)
    log.info(f"replaying [bold]{trial}[/] from {path} ({len(q) / fs:.1f} s, loops)")
    d.qpos[:] = q[0]
    mujoco.mj_forward(m, d)

    def step(m, d):
        d.time += m.opt.timestep
        k = int(d.time * fs) % len(q)
        d.qpos[:] = q[k]
        d.act[:] = act[k]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

    def render(scene):
        scene.update_from_mjdata(d)
        overlay.update(d, d.time, scene._scene_offset)   # camera-tracking offset of the mjviser scene

    return step, render


def run(mode="replay", shard=None, trial=None, emg_window=4.0, port=8080):
    m, d = load()
    server, render = viser_server(port), None
    if mode == "live":
        step = live(m, d)
    elif mode == "demo":
        step = demo(m, d)
    elif mode == "replay":
        step, render = replay(m, d, server, shard, trial or None, emg_window)
    else:
        raise ValueError(f"view.mode must be replay, demo or live, not {mode!r}")
    log.info(f"viewer ({mode}): [link=http://localhost:{port}]http://localhost:{port}[/link]  (Ctrl+C to stop)")
    Viewer(m, d, step_fn=step, render_fn=render, server=server).run()
