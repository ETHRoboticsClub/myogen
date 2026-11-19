from __future__ import annotations

import colorsys
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from ..recording import Recording
from ..simulation.geometry import ForearmGeometry
from ..simulation.musculoskeletal import ForearmModel

if TYPE_CHECKING:
    import viser

FRAME_PERIOD = 1 / 30   # s between scene updates (bones, forearm frame)
MUSCLE_PERIOD = 1 / 20  # s between muscle-surface updates (vertices and colours of 29 meshes, ~6 ms)
PLOT_PERIOD = 0.1       # s between EMG plot / electrode colour updates
PLOT_BINS = 400         # min/max bins across the EMG plot window
SPEEDS = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0}
SKIP_GEOMS = ("clavicle_r", "scapula_r")        # the shoulder girdle: off screen, it only hides the arm
FOREARM_BONES = ("radius_r", "ulna_r")
BONE_RGB = (236, 230, 214)
TISSUE = np.array([0.78, 0.50, 0.47])           # an idle muscle

log = logging.getLogger(__name__)


def replay(recording: Recording | str | Path, port: int = 8080, emg_window: float = 4.0) -> None:
    """Loop a Recording (or a recording file) in the web viewer until Ctrl+C."""
    import viser
    recording = recording if isinstance(recording, Recording) else Recording.load(recording)
    server = viser.ViserServer(port=port, verbose=False)
    view = ReplayView(server, ForearmModel(), recording, emg_window)
    log.info(f"replaying {recording.duration:.1f} s: http://localhost:{port}  (Ctrl+C to stop)")
    view.run()


class ReplayView:
    """The arm (the model's bones, the atlas muscle surfaces on its forearm, the other muscles' paths), the recording
    setup and the EMG panel, all from the recording's joint angles. run() plays it; update(t) draws time t."""

    def __init__(self, server: viser.ViserServer, model: ForearmModel, recording: Recording,
                 emg_window: float = 4.0) -> None:
        self.server, self.model, self.rec = server, model, recording
        sub, electrodes = recording.subject, recording.electrodes
        self.q, self.exc = recording.labels["q"], recording.labels["excitation"]
        self.fs = recording.fs_labels
        self.duration = len(self.q) / self.fs
        self.t, self.playing, self.speed = 0.0, True, 1.0
        self._last_plot = self._last_muscles = 0.0
        self._prepare_emg(recording.emg.astype(np.float32), recording.fs, emg_window)
        m, self.d = model.m, model.d

        # The muscles in the subject's cross-section, as the EMG model sees them.
        self.geometry = ForearmGeometry(model, sub)
        C = len(electrodes)
        self.chan_rgb = np.array([colorsys.hsv_to_rgb(k / C, 0.85, 1.0) for k in range(C)])
        self.owner = np.repeat(np.arange(C), 2)                     # the channel of each pole

        sc = server.scene
        sc.configure_environment_map("studio", background=False, environment_intensity=0.4)
        self._add_gravity_and_armrest(sc)                           # also sets the pose to the first frame
        sc.add_light_directional("/key", color=(255, 250, 240), intensity=1.8, cast_shadow=True,
                                 wxyz=_look_rotation(self.look - self.eye + 0.3 * self.down))

        # The model's bones, moved every frame; its radius and ulna can give way to the atlas's.
        self.bones: dict[int, Any] = {}
        for g in range(m.ngeom):
            name = m.geom(g).name
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and name not in SKIP_GEOMS:
                self.bones[g] = sc.add_mesh_simple(f"/bones/{name}", *_geom_mesh(m, g), color=BONE_RGB)
        self.model_forearm = [h for g, h in self.bones.items() if m.geom(g).name in FOREARM_BONES]

        # Everything in the forearm frame (z from the elbow along the forearm).
        self.forearm = sc.add_frame("/forearm", show_axes=False)
        self.atlas_bones = [sc.add_mesh_simple(f"/forearm/atlas_bones/{n}", v.astype(np.float32),
                                               f.astype(np.uint32), color=BONE_RGB, visible=False)
                            for n, v, f in self.geometry.bone_surfaces()]
        skin, z_end = self.geometry.skin_radius, sub.residual_length * model.forearm_length
        v, f = _frustum(skin(0.0), skin(z_end), z_end)
        self.skin = sc.add_mesh_simple("/forearm/skin", v, f, color=(230, 190, 160), opacity=0.12,
                                       side="double", cast_shadow=False, receive_shadow=False)
        poles = self.geometry.poles(electrodes)                     # (C, 2 or 1, 3) in the forearm frame
        self.electrodes = sc.add_point_cloud("/forearm/electrodes", poles.reshape(-1, 3).astype(np.float32),
                                             _rgb(self.chan_rgb[self.owner]), point_size=0.008, point_shape="circle")
        self.pairs = ([sc.add_line_segments("/forearm/bipolar", poles, colors=(40, 40, 40), thickness=3.0,
                                            thickness_units="screen")] if poles.shape[1] == 2 else [])
        self.labels = [sc.add_label(f"/forearm/labels/{n}", n, position=p.mean(0) * [1.18, 1.18, 1.0],
                                    font_screen_scale=0.7) for n, p in zip(electrodes.names, poles)]

        # Atlas muscles as surfaces; the others (upper arm, hand intrinsics) as their paths.
        surfaces = self.geometry.muscle_surfaces(self.d)
        self.anatomy_rgb = {mu: np.array(colorsys.hsv_to_rgb((k * 0.618) % 1, 0.6, 0.8))
                            for k, (mu, _, _) in enumerate(surfaces)}
        self.muscles = {mu: sc.add_mesh_simple(f"/forearm/muscles/{model.muscle_names[mu]}", v.astype(np.float32),
                                               f.astype(np.uint32), color=_rgb(TISSUE))
                        for mu, v, f in surfaces}
        self.paths: dict[int, Any] = {}
        for i in range(model.nu):
            if i in self.muscles or m.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON:
                continue
            seg = self._path(i)
            if seg is not None:
                self.paths[i] = sc.add_line_segments(f"/paths/{model.muscle_names[i]}", seg, colors=(200, 70, 60),
                                                     thickness=2.0, thickness_units="screen")
        self._gui()
        self._emg_plot(electrodes.names)
        self._camera_set: set[int] = set()

    # -- setup ---------------------------------------------------------------------------------------------------

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
        q0 = self.q[0]
        model.prepare(sub, q0, q0)                                  # sets the (armrest-projected) gravity
        gdir = model.m.opt.gravity / np.linalg.norm(model.m.opt.gravity)
        self.down = gdir
        model.set_pose(q0)
        L = model.forearm_length
        sc.set_up_direction(tuple(-gdir))
        elbow, axes = model.forearm_frame()
        tip = elbow - 0.18 * gdir + 0.04 * axes[2]
        sc.add_arrows("/gravity", np.stack([tip, tip + 0.12 * gdir])[None], colors=(60, 120, 255),
                      shaft_radius=0.003, head_radius=0.008, head_length=0.02)
        sc.add_label("/gravity_label", "g", position=tip + 0.06 * gdir + 0.015 * axes[0])
        self.look = elbow + 0.6 * L * axes[2]                       # mid-forearm, seen from above and to the side
        side = np.cross(gdir, axes[2])
        self.eye = self.look - 0.22 * gdir + 0.36 * side / np.linalg.norm(side)
        if sub.armrest:
            import viser.transforms as vtf
            x = axes[2] - (axes[2] @ gdir) * gdir
            x /= np.linalg.norm(x)
            R = np.stack([x, np.cross(-gdir, x), -gdir], 1)
            centre = elbow + 0.5 * L * x + (self.geometry.skin_radius(0.5 * L) + 0.004) * gdir
            sc.add_box("/armrest", color=(120, 110, 100), dimensions=(1.2 * L, 0.12, 0.006),
                       position=centre, wxyz=vtf.SO3.from_matrix(R).wxyz, opacity=0.6)

    def _gui(self) -> None:
        rec, sub, el = self.rec, self.rec.subject, self.rec.electrodes
        gui = self.server.gui
        with gui.add_folder("Playback"):
            self.play_button = gui.add_button("Pause")
            self.play_button.on_click(lambda _: self._toggle_play())
            self.speed_choice = gui.add_dropdown("Speed", tuple(SPEEDS), initial_value="1x")
            self.speed_choice.on_update(self._set_speed)
            self.time_slider = gui.add_slider("Time (s)", 0.0, round(self.duration, 2), 0.01, 0.0)
            self.time_slider.on_update(self._scrub)
        with gui.add_folder("Recording"):
            gui.add_markdown(
                f"**{el.device}** · {len(el)} electrodes · {rec.duration:.1f} s  \n"
                f"skin r {sub.forearm_radius_mm:.0f} mm, fat {sub.fat_mm:.1f} mm · "
                f"{'amputee (' + format(sub.residual_length, '.0%') + ')' if sub.amputee else 'intact'} · "
                f"{'armrest' if sub.armrest else 'arm held up'}  \n"
                f"tracking RMSE {rec.qc['rmse_deg']:.2f}° · QC {'pass' if rec.qc['pass'] else 'FAIL'}")
        with gui.add_folder("Show"):
            layers: list[tuple[str, list[Any], bool]] = [
                ("Skin", [self.skin], True), ("Electrodes", [self.electrodes, *self.pairs], True),
                ("Channel labels", list(self.labels), False), ("Muscles", list(self.muscles.values()), True),
                ("Other muscles (paths)", list(self.paths.values()), True)]
            for name, handles, on in layers:
                box = gui.add_checkbox(name, initial_value=on)
                _set_visible(handles, on)
                box.on_update(_toggle(box, handles))
            self.forearm_bones = gui.add_dropdown("Forearm bones", ("model", "atlas", "both", "none"))
            self.forearm_bones.on_update(lambda _: self._show_forearm_bones())
            self.muscle_colour = gui.add_dropdown("Muscle colour", ("activity", "anatomy"))
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

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.label = "Pause" if self.playing else "Play"

    def _set_speed(self, _: Any) -> None:
        self.speed = SPEEDS[self.speed_choice.value]

    def _scrub(self, event: Any) -> None:
        if event.client is not None:                                # a viewer moved it (not the playback)
            self.t = float(self.time_slider.value)
            self._last_muscles = self._last_plot = 0.0

    def _show_forearm_bones(self) -> None:
        mode = self.forearm_bones.value
        _set_visible(self.model_forearm, mode in ("model", "both"))
        _set_visible(self.atlas_bones, mode in ("atlas", "both"))

    # -- drawing -------------------------------------------------------------------------------------------------

    def run(self) -> None:
        """Play the recording in a loop until Ctrl+C."""
        last = time.perf_counter()
        while True:
            now = time.perf_counter()
            if self.playing:
                self.t = (self.t + (now - last) * self.speed) % self.duration
            last = now
            self.update(self.t)
            time.sleep(max(0.0, FRAME_PERIOD - (time.perf_counter() - now)))

    def update(self, t: float) -> None:
        """Draw replay time t (s)."""
        import viser.transforms as vtf
        m, d = self.model.m, self.d
        k = int(t * self.fs) % len(self.q)
        d.qpos[:] = self.q[k]
        d.qvel[:] = 0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_tendon(m, d)                                      # the paths' wrap points
        for cid, client in self.server.get_clients().items():       # place the camera for new clients
            if cid not in self._camera_set:
                self._camera_set.add(cid)
                with client.atomic():
                    client.camera.position, client.camera.look_at = tuple(self.eye), tuple(self.look)
        elbow, axes = self.model.forearm_frame(d)
        now = time.perf_counter()
        with self.server.atomic():
            for g, h in self.bones.items():
                h.position = d.geom_xpos[g]
                h.wxyz = vtf.SO3.from_matrix(d.geom_xmat[g].reshape(3, 3)).wxyz
            self.forearm.position = elbow
            self.forearm.wxyz = vtf.SO3.from_matrix(axes.T).wxyz
            if now - self._last_muscles > MUSCLE_PERIOD:
                self._last_muscles = now
                self._draw_muscles(self.exc[k])
            if now - self._last_plot > PLOT_PERIOD and self.electrodes.visible:
                rgb = self.chan_rgb[self.owner]
                if self.color_mode.value == "EMG activity":
                    a = self.env[int(t * self.fs_emg) % len(self.env)][self.owner, None]
                    rgb = (0.25 + 0.75 * a) * rgb + 0.15 * (1 - a)
                self.electrodes.colors = _rgb(rgb)
        if now - self._last_plot > PLOT_PERIOD:
            self._last_plot = now
            self.plot.data = self._emg_window(t)
            if self.playing:
                self.time_slider.value = round(t, 2)        # type: ignore[misc]

    def _draw_muscles(self, exc: np.ndarray) -> None:
        activity = self.muscle_colour.value == "activity"
        if any(h.visible for h in self.muscles.values()):
            for mu, v, _ in self.geometry.muscle_surfaces(self.d):
                h = self.muscles[mu]
                h.vertices = v.astype(np.float32)
                h.color = tuple(_rgb(_activity(exc[mu]) if activity else self.anatomy_rgb[mu]))
        for i, h in self.paths.items():
            seg = self._path(i)
            if seg is not None and h.visible:
                h.points = seg
                h.colors = np.broadcast_to(_rgb(_activity(exc[i])), seg.shape).copy()

    def _path(self, muscle: int) -> np.ndarray | None:
        """(N, 2, 3) segments of a tendon-driven muscle's path now (world frame), or None."""
        t = self.model.m.actuator_trnid[muscle, 0]
        adr, n = self.d.ten_wrapadr[t], self.d.ten_wrapnum[t]
        p = self.d.wrap_xpos.reshape(-1, 3)[adr:adr + n]
        p = p[np.any(p != 0, axis=1)]
        return np.stack([p[:-1], p[1:]], 1).astype(np.float32) if len(p) > 1 else None

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


def _toggle(box: Any, handles: list[Any]) -> Any:
    """A checkbox callback showing or hiding these handles."""
    return lambda _: _set_visible(handles, box.value)


def _set_visible(handles: list[Any], visible: bool) -> None:
    for h in handles:
        h.visible = visible


def _rgb(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _activity(a: float) -> np.ndarray:
    """Idle tissue -> red -> yellow as the excitation goes 0 -> 0.5 -> 1."""
    stops = np.array([TISSUE, [0.95, 0.18, 0.10], [1.0, 0.85, 0.25]])
    a = float(np.clip(a, 0, 1))
    return np.array([np.interp(a, (0.0, 0.5, 1.0), stops[:, c]) for c in range(3)])


def _geom_mesh(m: mujoco.MjModel, g: int) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of a mesh geom, in the geom's frame."""
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    return v.astype(np.float32), f.astype(np.uint32)


def _look_rotation(direction: np.ndarray) -> np.ndarray:
    """wxyz of a frame whose -z looks along direction (directional lights shine along their -z)."""
    import viser.transforms as vtf
    z = -direction / np.linalg.norm(direction)
    x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return vtf.SO3.from_matrix(np.stack([x, np.cross(z, x), z], 1)).wxyz


def _frustum(r0: float, r1: float, length: float, n: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Open tapered tube along +z: (vertices, faces)."""
    a = 2 * np.pi * np.arange(n) / n
    ring = np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1)
    v = np.concatenate([ring * [r0, r0, 0], ring * [r1, r1, 0] + [0, 0, length]])
    i = np.arange(n)
    j = (i + 1) % n
    f = np.concatenate([np.stack([i, j, n + i], 1), np.stack([j, n + j, n + i], 1)])
    return v.astype(np.float32), f.astype(np.uint32)
