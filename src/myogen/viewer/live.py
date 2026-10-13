"""Live view of a running generation (generate.live_view: true): worker 0 streams its simulation to the viewer.

The arm moves as it is simulated, with the trial's recording setup; EMG is synthesised after each trial, so it is
not shown here (replay finished trials with `myogen view`).
"""
import logging
import time

import mujoco
from mjviser.scene import ViserMujocoScene

from ..body.model import load
from ..log import viser_server
from .overlay import TrialOverlay

log = logging.getLogger(__name__)
FRAME = 1 / 30


class LiveView:
    def __init__(self, port, fs_label=100):
        self.m, self.d = load()
        self.fs_label = fs_label
        self.server = viser_server(port)
        self.scene = ViserMujocoScene(self.server, self.m, 1)
        self.scene.show_tendons = True
        with self.server.gui.add_folder("Live generation"):
            self.status = self.server.gui.add_markdown("waiting for the first trial ...")
        self.overlay = self.exc = None
        self.name, self.progress, self.t, self.dirty, self.last = "", "", 0.0, False, 0.0
        log.info(f"live view: [link=http://localhost:{port}]http://localhost:{port}[/link] (worker 0)")

    def handle(self, ev, done, total):
        kind = ev[0]
        if kind == "setup":
            _, self.name, setup = ev
            if self.overlay is not None:
                self.overlay.remove()
            self.d.qpos[:] = setup["q0"]
            mujoco.mj_forward(self.m, self.d)
            self.overlay = TrialOverlay(self.server, self.m, self.d, setup["subject_params"], setup["session_params"],
                                        setup["meta"], setup["q0"])
        elif kind == "state":
            _, qpos, act, self.exc, self.t = ev
            self.d.qpos[:], self.d.act[:] = qpos, act
            self.dirty = True
        self.progress = f"{done} / {total} trials done"

    def render(self):
        now = time.perf_counter()
        if not self.dirty or now - self.last < FRAME:
            return
        self.last, self.dirty = now, False
        mujoco.mj_forward(self.m, self.d)
        self.scene.update_from_mjdata(self.d)
        if self.overlay is not None:
            self.overlay.update(self.d, self.t, self.scene._scene_offset, exc=self.exc)
        self.status.content = f"**{self.name}** · {self.t:.1f} s  \n{self.progress}"

    def finished(self, n):
        self.status.content = f"generation finished: {n} trials. Ctrl+C to close the viewer."
        log.info("generation finished; the live view stays open until Ctrl+C")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()
