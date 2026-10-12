"""3D overlay of a generated trial's recording setup in the web viewer (`myogen view`, replay mode).

Everything is drawn in the forearm frame (src.model.forearm_frame: elbow origin, z along the forearm, attached to
the ulna), the same frame the EMG models use, so it rides with the arm during replay:

  * skin: the subject's tapered forearm (forearm_radius, taper), cut at the residual length for amputees
  * electrodes: the stored per-trial positions (analytic model: `electrodes_m`; BioMime: the bipolar poles in
    BioMime grid coordinates, mapped back with the anchor-fitted angle offset), coloured by channel or by EMG
  * muscle sources: the analytic model's belly source points mapped into this subject's cross-section,
    coloured by neural excitation (for BioMime trials they show where the muscles are, not the MU territories)
  * gravity arrow (the camera's up direction is set against it) and, for supported sessions, the armrest
  * a scrolling plot of every EMG channel (stacked, each normalised to its 99th percentile, min/max decimated),
    in the same colours as the electrodes

It is built from the trial's parameters, so it works for a stored trial (TrialOverlay.from_trial, replay) and for
one that is still being simulated (generate.live_view: no EMG yet, excitations passed to update()).
"""
import colorsys
import json
import time
import weakref

import mujoco
import numpy as np
import viser.transforms as vtf
import viser.uplot as uplot
from matplotlib import colormaps
from scipy import signal

from .emg import EMGGeometry
from .model import Anatomy, forearm_frame
from .subject import ElectrodeLayout, Session, Subject

ENV_SMOOTH_S = 0.1   # moving-average window of the EMG envelope used to colour electrodes
PLOT_BINS = 400      # min/max bins across the EMG plot window
PLOT_PERIOD = 0.1    # s between EMG plot updates (wall clock)
_CAMERA_SET = weakref.WeakKeyDictionary()   # server -> client ids whose camera was already placed


def _params(p):
    return json.loads(p) if isinstance(p, str) else dict(p)


def _subject(params):
    p = _params(params)
    p["layout"] = ElectrodeLayout(**{**p["layout"], "targets": tuple(p["layout"]["targets"])})
    return Subject(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()})


def _session(params):
    return Session(**{k: np.asarray(v) if isinstance(v, list) else v for k, v in _params(params).items()})


def _channel_names(layout):
    per_ring = (layout.n_channels - len(layout.targets)) // layout.rings
    return ([f"ring{r}_{k}" for r in range(layout.rings) for k in range(per_ring)]
            + [f"target_{t}" for t in layout.targets])


def _envelope(emg, fs):
    """(T, C) activity in [0, 1]: smoothed |emg| normalised per channel by its 99th percentile."""
    x = np.abs(emg.astype(np.float32))
    n = max(1, int(ENV_SMOOTH_S * fs))
    x = signal.lfilter(np.ones(n) / n, 1.0, x, axis=0)
    return np.clip(x / np.maximum(np.percentile(x, 99, axis=0), 1e-9), 0, 1)


def _frustum(r0, r1, length, n=48):
    """Open tapered tube along +z (vertices, faces)."""
    a = 2 * np.pi * np.arange(n) / n
    ring = np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1)
    v = np.concatenate([ring * [r0, r0, 0], ring * [r1, r1, 0] + [0, 0, length]])
    i = np.arange(n)
    j = (i + 1) % n
    f = np.concatenate([np.stack([i, j, n + i], 1), np.stack([j, n + j, n + i], 1)])
    return v.astype(np.float32), f.astype(np.uint32)


class TrialOverlay:
    @classmethod
    def from_trial(cls, server, m, d, g, q0, emg_window=4.0):
        """g: a stored trial's HDF5 group."""
        return cls(server, m, d, g.attrs["subject_params"], g.attrs["session_params"], json.loads(g.attrs["meta"]),
                   q0, exc=g["labels/exc"][:], fs_label=float(g.attrs["fs_label"]), emg=g["emg"][:],
                   fs_emg=float(g.attrs["fs_emg"]), emg_window=emg_window)

    def __init__(self, server, m, d, subject_params, session_params, meta, q0, exc=None, fs_label=100.0, emg=None,
                 fs_emg=None, emg_window=4.0):
        """subject_params / session_params: dicts or JSON (as stored per trial); meta: trial meta with the electrode
        positions; q0: first posture (targeted electrodes / sources are set up there); exc: (T, nu) excitations at
        fs_label (or pass them to update()); emg: (T, C) at fs_emg for electrode colours and the scrolling plot."""
        self.server, self.m = server, m
        self.sub, self.ses = _subject(subject_params), _session(session_params)
        self.anat = Anatomy.from_model(m, d)
        L = self.anat.forearm_length
        self.exc, self.fs_label, self.fs_emg = exc, fs_label, fs_emg
        self.env = self.plot = None
        if emg is not None:
            emg = emg.astype(np.float32)
            self.env = _envelope(emg, fs_emg)
            # Stacked traces: channel 0 on top, each scaled to +-0.5 of its 99th percentile.
            C = emg.shape[1]
            self.stacked = emg / (2 * np.maximum(np.percentile(np.abs(emg), 99, axis=0), 1e-9)) + np.arange(C)[::-1]
            self.win = min(int(emg_window * fs_emg), len(emg)) // PLOT_BINS * PLOT_BINS or len(emg)
        self._last_plot = 0.0

        d0 = mujoco.MjData(m)
        d0.qpos[:] = q0
        mujoco.mj_forward(m, d0)
        self.geom = EMGGeometry(m, d0, self.anat, self.sub, self.ses)
        skin = self.geom.skin_radius

        # Electrode poles (P, 3) in the forearm frame, the channel each belongs to, and a label anchor per channel.
        if "electrodes_m" in meta:
            poles = np.asarray(meta["electrodes_m"])
            names = _channel_names(self.sub.layout)
            owner = np.arange(len(poles))
            pairs = np.zeros((0, 2, 3))
        else:
            from .biomime.anatomy import fit_angle_transform
            ref = mujoco.MjData(m)                      # the MUAP bank's territories were fitted at the model's default pose
            mujoco.mj_forward(m, ref)
            offset, _ = fit_angle_transform(m, ref, Anatomy.from_model(m, ref))
            names = [ch["name"] for ch in meta["electrodes"]]
            pts, owner = [], []
            for c, ch in enumerate(meta["electrodes"]):
                for a, zf, _ in ch["poles"]:
                    th, z = 2 * np.pi * a - offset, zf * L
                    pts.append([skin(z) * np.cos(th), skin(z) * np.sin(th), z])
                    owner.append(c)
            poles, owner = np.asarray(pts), np.asarray(owner)
            pairs = np.stack([poles[owner == c][:2] for c in range(len(names)) if (owner == c).sum() >= 2])
        self.owner = owner
        C = len(names)
        self.chan_rgb = np.array([colorsys.hsv_to_rgb(k / C, 0.85, 1.0) for k in range(C)])
        centre = np.stack([poles[owner == c].mean(0) for c in range(C)])

        # Scene: /experiment follows the viewer's camera-tracking offset, /experiment/forearm the forearm frame.
        sc = server.scene
        self.root = sc.add_frame("/experiment", show_axes=False)
        self.forearm = sc.add_frame("/experiment/forearm", show_axes=False)
        z_end = self.sub.residual_length * L
        v, f = _frustum(skin(0.0), skin(z_end), z_end)
        self.skin = sc.add_mesh_simple("/experiment/forearm/skin", v, f, color=(230, 190, 160), opacity=0.25,
                                       side="double", cast_shadow=False, receive_shadow=False)
        self.electrodes = sc.add_point_cloud("/experiment/forearm/electrodes", poles.astype(np.float32),
                                             self._rgb(self.chan_rgb[owner]), point_size=0.008,
                                             point_shape="circle")
        self.pairs = (sc.add_line_segments("/experiment/forearm/bipolar", pairs, colors=(40, 40, 40),
                                           line_width=3.0) if len(pairs) else None)
        self.labels = [sc.add_label(f"/experiment/forearm/labels/{n}", n,
                                    position=c * [1.18, 1.18, 1.0], font_screen_scale=0.7)
                       for n, c in zip(names, centre)]
        src = self.geom.sources(d0)
        self.alive = ((src[..., 2] < z_end) & (src[..., 2] > -0.02)).reshape(-1)   # as in EMGGeometry.gains
        self.cmap = colormaps["inferno"]
        self.sources = sc.add_point_cloud("/experiment/forearm/sources", src.reshape(-1, 3)[self.alive],
                                          np.zeros((int(self.alive.sum()), 3), np.uint8), point_size=0.004,
                                          point_shape="circle")

        # Gravity (stored after generation's armrest projection) and the armrest for supported sessions.
        gdir = np.asarray(self.ses.gravity_dir, float)
        sc.set_up_direction(tuple(-gdir))
        elbow, axes = forearm_frame(m, d0)
        tip = elbow - 0.18 * gdir + 0.04 * axes[2]
        sc.add_arrows("/experiment/gravity", np.stack([tip, tip + 0.12 * gdir])[None], colors=(60, 120, 255),
                      shaft_radius=0.003, head_radius=0.008, head_length=0.02)
        sc.add_label("/experiment/gravity_label", "g", position=tip + 0.06 * gdir + 0.015 * axes[0])
        # Camera: looking at the mid-forearm from above and to the side (gravity down).
        self.look = elbow + 0.5 * L * axes[2]
        self.eye = self.look - 0.4 * gdir + 0.45 * np.cross(gdir, axes[2])
        if self.ses.arm_support > 0.5:
            x = axes[2] - (axes[2] @ gdir) * gdir
            x /= np.linalg.norm(x)
            R = np.stack([x, np.cross(-gdir, x), -gdir], 1)
            ctr = elbow + 0.5 * L * x + (skin(0.5 * L) + 0.004) * gdir
            sc.add_box("/experiment/armrest", color=(120, 110, 100), dimensions=(1.2 * L, 0.12, 0.006),
                       position=ctr, wxyz=vtf.SO3.from_matrix(R).wxyz, opacity=0.6)

        self.folders = [self._gui(meta, C)]
        if self.env is not None:
            self.folders.append(self._emg_plot(names))

    def remove(self):
        """Take everything this overlay added off the scene and the panel."""
        self.root.remove()
        for f in self.folders:
            f.remove()

    def _rgb(self, x):
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def _gui(self, meta, C):
        sub, ses, lay = self.sub, self.ses, self.sub.layout
        gui = self.server.gui
        with gui.add_folder("Experiment") as folder:
            gui.add_markdown(
                f"**{lay.device}** · {C} ch · {lay.rings} ring(s){' + ' + ', '.join(lay.targets) if lay.targets else ''}  \n"
                f"EMG: {meta.get('emg_model', 'analytic')} · motion: {meta.get('motion', '?')}  \n"
                f"ring at {lay.level + ses.level_shift:.0%} of forearm, rotation "
                f"{np.degrees(lay.rotation + ses.rotation_shift):.0f}°  \n"
                f"skin r {sub.forearm_radius * 1e3:.0f} mm, fat {sub.fat * 1e3:.1f} mm, decay {sub.decay * 1e3:.1f} mm  \n"
                f"{'amputee (residual ' + format(sub.residual_length, '.0%') + ')' if sub.amputee else 'intact'} · "
                f"{'armrest' if ses.arm_support > 0.5 else 'free arm'} ({ses.arm_support:.2f}) · "
                f"elbow {np.degrees(ses.elbow_angle):.0f}°")
            for name, handles in [("Skin", [self.skin]), ("Electrodes", [self.electrodes, self.pairs]),
                                  ("Channel labels", self.labels), ("Muscle sources", [self.sources])]:
                cb = gui.add_checkbox(name, initial_value=name != "Channel labels")
                for h in handles:
                    if h:
                        h.visible = cb.value
                cb.on_update(lambda _, cb=cb, hs=handles: [setattr(h, "visible", cb.value) for h in hs if h])
            modes = ("EMG activity", "channel") if self.env is not None else ("channel",)
            self.color_mode = gui.add_dropdown("Electrode colour", modes)
        return folder

    def _emg_plot(self, names):
        hexes = ["#%02x%02x%02x" % tuple(self._rgb(c)) for c in self.chan_rgb]
        C = len(names)
        with self.server.gui.add_folder("EMG") as folder:
            self.plot = self.server.gui.add_uplot(
                self._emg_window(0.0),
                ({"label": "s"}, *({"label": n, "stroke": h, "width": 1} for n, h in zip(names, hexes))),
                scales={"x": {"time": False, "auto": True}, "y": {"auto": False, "range": (-0.8, C - 0.2)}},
                axes=({}, {"show": False}), cursor={"show": False}, legend={"show": True, "live": False},
                height=40 + 22 * C, aspect=0.8)
        return folder

    def _emg_window(self, t):
        """Last `win` samples before t (wrapping like the replay), min/max decimated: (x, *channels)."""
        k1 = int(t * self.fs_emg)
        idx = np.arange(k1 - self.win, k1)
        x = (idx - k1) / self.fs_emg
        seg = np.take(self.stacked, idx, axis=0, mode="wrap")
        n = self.win // PLOT_BINS
        if n >= 2:
            seg = seg.reshape(PLOT_BINS, n, -1)
            seg = np.stack([seg.min(1), seg.max(1)], 1).reshape(2 * PLOT_BINS, -1)
            x = x.reshape(PLOT_BINS, n)[:, [0, -1]].reshape(-1)
        return (x.astype(np.float64), *seg.T.astype(np.float64))

    def update(self, d, t, offset, exc=None):
        """t: trial time (s); offset: the viewer's camera-tracking scene offset; exc: current excitations
        (default: the stored ones at t)."""
        # New clients: set the camera here, after mjviser's own connect handler has placed it.
        seen = _CAMERA_SET.setdefault(self.server, set())
        for cid, client in self.server.get_clients().items():
            if cid not in seen:
                seen.add(cid)
                client.camera.position, client.camera.look_at = self.eye + offset, self.look + offset
        elbow, axes = forearm_frame(self.m, d)
        with self.server.atomic():
            self.root.position = offset
            self.forearm.position = elbow
            self.forearm.wxyz = vtf.SO3.from_matrix(axes.T).wxyz
            if exc is None and self.exc is not None:
                exc = self.exc[int(t * self.fs_label) % len(self.exc)]
            if self.sources.visible and exc is not None:
                self.sources.points = self.geom.sources(d).reshape(-1, 3)[self.alive].astype(np.float32)
                drive = np.repeat(exc, self.geom.K)[self.alive]
                self.sources.colors = self._rgb(self.cmap(0.15 + 0.85 * drive)[:, :3])
            if self.electrodes.visible:
                if self.color_mode.value == "channel" or self.env is None:
                    rgb = self.chan_rgb[self.owner]
                else:
                    a = self.env[int(t * self.fs_emg) % len(self.env)][self.owner, None]
                    rgb = (0.25 + 0.75 * a) * self.chan_rgb[self.owner] + 0.15 * (1 - a)
                self.electrodes.colors = self._rgb(rgb)
        now = time.perf_counter()
        if self.plot is not None and now - self._last_plot > PLOT_PERIOD:
            self._last_plot = now
            self.plot.data = self._emg_window(t)
