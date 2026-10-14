from __future__ import annotations

import colorsys
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from ..recording import Recording
from ..simulation.emg import EMGModel
from ..simulation.musculoskeletal import ForearmModel

if TYPE_CHECKING:
    import viser

PLOT_BINS = 400       # min/max bins across the EMG plot window
PLOT_PERIOD = 0.1     # s between EMG plot / electrode colour updates (wall clock)
SOURCES_PERIOD = 0.25  # s between muscle-belly updates (re-sending the point cloud every frame stalls the viewer)

log = logging.getLogger(__name__)


def replay(recording: Recording | str | Path, port: int = 8080, emg_window: float = 4.0) -> None:
    """Loop a Recording (or a recording file) in the web viewer until Ctrl+C."""
    import viser
    from mjviser import Viewer
    recording = recording if isinstance(recording, Recording) else Recording.load(recording)
    model = ForearmModel()
    m, d = model.m, model.d
    q = recording.labels["q"]
    fs = recording.fs_labels
    server = viser.ViserServer(port=port, verbose=False)
    view = ReplayView(server, model, recording, emg_window)

    def step(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        d.time += m.opt.timestep
        d.qpos[:] = q[int(d.time * fs) % len(q)]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

    def render(scene: Any) -> None:
        scene.update_from_mjdata(d)
        view.update(d, d.time, scene._scene_offset)     # follow mjviser's camera-tracking offset

    model.set_pose(q[0])
    log.info(f"replaying {recording.duration:.1f} s: http://localhost:{port}  (Ctrl+C to stop)")
    Viewer(m, d, step_fn=step, render_fn=render, server=server).run()


class ReplayView:
    """The recording setup drawn over the arm and the EMG panel; update() once per rendered frame."""

    def __init__(self, server: viser.ViserServer, model: ForearmModel, recording: Recording,
                 emg_window: float = 4.0) -> None:
        self.server, self.model, self.rec = server, model, recording
        sub, electrodes = recording.subject, recording.electrodes
        self.exc = recording.labels["excitation"]
        self._last_plot = self._last_sources = self._last_colors = 0.0
        self._prepare_emg(recording.emg.astype(np.float32), recording.fs, emg_window)

        # Muscle bellies mapped into the subject's cross-section, as the EMG model sees them.
        self.bellies = EMGModel(model, sub, electrodes)
        C = len(electrodes)
        self.chan_rgb = np.array([colorsys.hsv_to_rgb(k / C, 0.85, 1.0) for k in range(C)])
        self.owner = np.repeat(np.arange(C), 2)                     # the channel of each pole

        sc = server.scene
        model.set_pose(recording.labels["q"][0])
        self.root = sc.add_frame("/setup", show_axes=False)       # follows the camera-tracking offset
        self.forearm = sc.add_frame("/setup/forearm", show_axes=False)
        skin, z_end = self.bellies.skin_radius, sub.residual_length * model.forearm_length
        v, f = _frustum(skin(0.0), skin(z_end), z_end)
        self.skin = sc.add_mesh_simple("/setup/forearm/skin", v, f, color=(230, 190, 160), opacity=0.25,
                                       side="double", cast_shadow=False, receive_shadow=False)
        poles = self.bellies.poles                                  # (C, 2, 3) in the forearm frame
        self.electrodes = sc.add_point_cloud("/setup/forearm/electrodes", poles.reshape(-1, 3).astype(np.float32),
                                             _rgb(self.chan_rgb[self.owner]), point_size=0.008, point_shape="circle")
        self.pairs = sc.add_line_segments("/setup/forearm/bipolar", poles, colors=(40, 40, 40), line_width=3.0)
        self.labels = [sc.add_label(f"/setup/forearm/labels/{n}", n, position=p.mean(0) * [1.18, 1.18, 1.0],
                                    font_screen_scale=0.7) for n, p in zip(electrodes.names, poles)]
        src = self.bellies.sources(model.d)
        self.alive = ((src[..., 2] < z_end) & (src[..., 2] > -self.bellies.params.elbow_margin_m)).reshape(-1)
        self.sources = sc.add_point_cloud("/setup/forearm/muscles", src.reshape(-1, 3)[self.alive],
                                          np.zeros((int(self.alive.sum()), 3), np.uint8), point_size=0.004,
                                          point_shape="circle")
        self._add_gravity_and_armrest(sc)
        self._gui()
        self._emg_plot(electrodes.names)
        self._camera_set: set[int] = set()

    def _prepare_emg(self, emg: np.ndarray, fs: float, emg_window: float) -> None:
        self.fs_emg = fs
        x = np.abs(emg)
        n = max(1, int(0.1 * fs))                                   # 100 ms envelope for the electrode colours
        x = np.apply_along_axis(lambda c: np.convolve(c, np.ones(n) / n, "same"), 0, x)
        self.env = np.clip(x / np.maximum(np.percentile(x, 99, axis=0), 1e-9), 0, 1)
        # Stacked traces: channel 0 on top, each scaled to +-0.5 of its 99th percentile.
        C = emg.shape[1]
        self.stacked = emg / (2 * np.maximum(np.percentile(np.abs(emg), 99, axis=0), 1e-9)) + np.arange(C)[::-1]
        self.win = min(int(emg_window * fs), len(emg)) // PLOT_BINS * PLOT_BINS or len(emg)

    def _add_gravity_and_armrest(self, sc: viser.SceneApi) -> None:
        """Gravity arrow (the camera's up is against it), camera placement, and the armrest if supported."""
        model, sub = self.model, self.rec.subject
        q0 = self.rec.labels["q"][0]
        model.prepare(sub, q0, q0)                                  # sets the (armrest-projected) gravity
        gdir = model.m.opt.gravity / np.linalg.norm(model.m.opt.gravity)
        model.set_pose(q0)
        L = model.forearm_length
        sc.set_up_direction(tuple(-gdir))
        elbow, axes = model.forearm_frame()
        tip = elbow - 0.18 * gdir + 0.04 * axes[2]
        sc.add_arrows("/setup/gravity", np.stack([tip, tip + 0.12 * gdir])[None], colors=(60, 120, 255),
                      shaft_radius=0.003, head_radius=0.008, head_length=0.02)
        sc.add_label("/setup/gravity_label", "g", position=tip + 0.06 * gdir + 0.015 * axes[0])
        self.look = elbow + 0.5 * L * axes[2]                       # mid-forearm, seen from above and to the side
        self.eye = self.look - 0.4 * gdir + 0.45 * np.cross(gdir, axes[2])
        if sub.armrest:
            import viser.transforms as vtf
            x = axes[2] - (axes[2] @ gdir) * gdir
            x /= np.linalg.norm(x)
            R = np.stack([x, np.cross(-gdir, x), -gdir], 1)
            centre = elbow + 0.5 * L * x + (self.bellies.skin_radius(0.5 * L) + 0.004) * gdir
            sc.add_box("/setup/armrest", color=(120, 110, 100), dimensions=(1.2 * L, 0.12, 0.006),
                       position=centre, wxyz=vtf.SO3.from_matrix(R).wxyz, opacity=0.6)

    def _gui(self) -> None:
        rec, sub, el = self.rec, self.rec.subject, self.rec.electrodes
        gui = self.server.gui
        with gui.add_folder("Recording"):
            gui.add_markdown(
                f"**{el.device}** · {len(el)} electrodes · {rec.duration:.1f} s  \n"
                f"skin r {sub.forearm_radius_mm:.0f} mm, fat {sub.fat_mm:.1f} mm · "
                f"{'amputee (' + format(sub.residual_length, '.0%') + ')' if sub.amputee else 'intact'} · "
                f"{'armrest' if sub.armrest else 'arm held up'}  \n"
                f"tracking RMSE {rec.qc['rmse_deg']:.2f}° · QC {'pass' if rec.qc['pass'] else 'FAIL'}")
            for name, handles in [("Skin", [self.skin]), ("Electrodes", [self.electrodes, self.pairs]),
                                  ("Channel labels", self.labels), ("Muscles", [self.sources])]:
                box = gui.add_checkbox(name, initial_value=name != "Channel labels")
                for h in handles:
                    h.visible = box.value
                box.on_update(lambda _, box=box, hs=handles: _set_visible(hs, box.value))
            self.color_mode = gui.add_dropdown("Electrode colour", ("EMG activity", "channel"))

    def _emg_plot(self, names: list[str]) -> None:
        hexes = ["#{:02x}{:02x}{:02x}".format(*_rgb(c)) for c in self.chan_rgb]
        C = len(names)
        with self.server.gui.add_folder("EMG"):
            self.plot = self.server.gui.add_uplot(
                self._emg_window(0.0),
                ({"label": "s"}, *({"label": n, "stroke": h, "width": 1} for n, h in zip(names, hexes))),
                scales={"x": {"time": False, "auto": True}, "y": {"auto": False, "range": (-0.8, C - 0.2)}},
                axes=({}, {"show": False}), cursor={"show": False}, legend={"show": True, "live": False},
                height=40 + 22 * C, aspect=0.8)

    def _emg_window(self, t: float) -> tuple[np.ndarray, ...]:
        """The `win` samples before t (wrapping like the replay), min/max decimated: (x, *channels)."""
        k1 = int(t * self.fs_emg)
        idx = np.arange(k1 - self.win, k1)
        x = (idx - k1) / self.fs_emg
        seg = np.take(self.stacked, idx, axis=0, mode="wrap")
        n = self.win // PLOT_BINS
        if n >= 2:
            seg = np.stack([seg.reshape(PLOT_BINS, n, -1).min(1), seg.reshape(PLOT_BINS, n, -1).max(1)], 1)
            seg = seg.reshape(2 * PLOT_BINS, -1)
            x = x.reshape(PLOT_BINS, n)[:, [0, -1]].reshape(-1)
        return (x.astype(np.float64), *seg.T.astype(np.float64))

    def update(self, d: mujoco.MjData, t: float, offset: np.ndarray) -> None:
        """t: replay time (s); offset: the viewer's camera-tracking scene offset."""
        import viser.transforms as vtf
        for cid, client in self.server.get_clients().items():       # place the camera for new clients
            if cid not in self._camera_set:
                self._camera_set.add(cid)
                client.camera.position, client.camera.look_at = self.eye + offset, self.look + offset
        elbow, axes = self.model.forearm_frame(d)
        now = time.perf_counter()
        with self.server.atomic():
            self.root.position = offset
            self.forearm.position = elbow
            self.forearm.wxyz = vtf.SO3.from_matrix(axes.T).wxyz
            if self.sources.visible and now - self._last_sources > SOURCES_PERIOD:
                self._last_sources = now
                exc = self.exc[int(t * self.rec.fs_labels) % len(self.exc)]
                drive = np.repeat(exc, self.bellies.n_sources)[self.alive]
                self.sources.points = self.bellies.sources(d).reshape(-1, 3)[self.alive].astype(np.float32)
                self.sources.colors = _rgb(_heat(drive))
            if self.electrodes.visible and now - self._last_colors > PLOT_PERIOD:
                self._last_colors = now
                rgb = self.chan_rgb[self.owner]
                if self.color_mode.value == "EMG activity":
                    a = self.env[int(t * self.fs_emg) % len(self.env)][self.owner, None]
                    rgb = (0.25 + 0.75 * a) * rgb + 0.15 * (1 - a)
                self.electrodes.colors = _rgb(rgb)
        if now - self._last_plot > PLOT_PERIOD:
            self._last_plot = now
            self.plot.data = self._emg_window(t)


def _set_visible(handles: list[Any], visible: bool) -> None:
    for h in handles:
        h.visible = visible


def _rgb(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _heat(x: np.ndarray) -> np.ndarray:
    """Black -> red -> yellow for x in [0, 1]."""
    x = 0.15 + 0.85 * np.clip(x, 0, 1)
    return np.stack([np.clip(2 * x, 0, 1), np.clip(2 * x - 0.8, 0, 1), np.clip(3 * x - 2.4, 0, 1)], axis=-1)


def _frustum(r0: float, r1: float, length: float, n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Open tapered tube along +z: (vertices, faces)."""
    a = 2 * np.pi * np.arange(n) / n
    ring = np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1)
    v = np.concatenate([ring * [r0, r0, 0], ring * [r1, r1, 0] + [0, 0, length]])
    i = np.arange(n)
    j = (i + 1) % n
    f = np.concatenate([np.stack([i, j, n + i], 1), np.stack([j, n + j, n + i], 1)])
    return v.astype(np.float32), f.astype(np.uint32)
