from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, overload

import mujoco
import numpy as np

from ..params import EMGParams
from .anatomy import TRACK_POINTS, Anatomy, AtlasPoints
from .motor_units import MotorUnits
from .musculoskeletal import R_MODEL_REF

if TYPE_CHECKING:
    from ..inputs.electrodes import Electrodes
    from ..inputs.subject import Subject
    from .muscle_activity import MuscleActivity
    from .musculoskeletal import ForearmModel


class EMGModel:
    """Clean sEMG at the electrodes from muscle activity (forearm frame: z along the forearm from the elbow,
    metres). The forearm muscles sit where the BodyParts3D atlas has them (simulation/anatomy.py); each is a
    motor-unit pool (MotorUnits) whose action potentials every electrode pole sees; bipolar = pole 1 - pole 2 delayed
    by the conduction time between them; monopolar = the electrode's single pole against a distant reference, or
    against the average of the electrodes."""

    def __init__(self, model: ForearmModel, subject: Subject, electrodes: Electrodes,
                 params: EMGParams | None = None) -> None:
        self.model, self.sub, self.params = model, subject, params or EMGParams()
        p = self.params
        self.decay, self.fs = p.decay_mm / 1000, p.fs
        self.data = mujoco.MjData(model.m)              # scratch state for the per-tick geometry
        self.fat = subject.fat_mm / 1000
        self.limb_length = subject.residual_length * model.forearm_length   # shorter for amputees
        self.monopolar, self.reference = electrodes.monopolar, electrodes.reference

        self.poles = self._pole_positions(electrodes)                       # (C, 2 or 1 poles, xyz)
        self.anatomy = Anatomy(model, subject)
        self._track = mujoco.MjData(model.m)            # scratch state for the bellies (bellies())

        # Delay (s) between the two poles: the along-fibre part of the pole spacing over the conduction velocity.
        along_fibre = np.array([e.ied_mm / 1000 * np.cos(np.radians(e.orientation_deg))
                                for e in electrodes.electrodes])
        self.pole_delay = along_fibre / p.conduction_velocity
        if p.motor_units.waveform == "volume_conductor":
            self.pole_delay = np.zeros_like(along_fibre)     # each pole's waveform already propagates

        self.units = MotorUnits(self, p.motor_units)

    def generate(self, activity: MuscleActivity, rng: np.random.Generator,
                 on_tick: Callable[[int, int], None] | None = None) -> np.ndarray:
        """(T * upsample, C) EMG in mV at self.fs, from the drive and joint angles at the control rate.
        on_tick(k, T) is called after every chunk of ticks."""
        pole_emg = self.units.pole_signals(activity.drive, activity.qpos, activity.fs, rng, on_tick=on_tick)
        if self.monopolar:
            emg = pole_emg - pole_emg.mean(axis=1, keepdims=True) if self.reference == "average" else pole_emg
            scale = self.params.monopolar_scale
        else:
            emg, scale = self._bipolar(pole_emg), self.params.bipolar_scale
        scale *= self.sub.gain_mv
        return (emg * scale).astype(np.float32)

    # -- geometry ----------------------------------------------------------------------------------------------

    @overload
    def skin_radius(self, z: float) -> float: ...
    @overload
    def skin_radius(self, z: np.ndarray) -> np.ndarray: ...

    def skin_radius(self, z: float | np.ndarray) -> float | np.ndarray:
        """Forearm radius (m) at distance z from the elbow; it tapers towards the wrist."""
        frac_along = np.clip(z / self.model.forearm_length, 0, 1)
        return self.sub.forearm_radius_mm / 1000 * (1 - self.sub.taper * frac_along)

    def _pole_positions(self, electrodes: Electrodes) -> np.ndarray:
        """(C, 2, 3) both poles of every electrode as points on the skin; (C, 1, 3) its centre if monopolar."""
        poles = []
        for e in electrodes.electrodes:
            pole_ends = e.poles(self.skin_radius)   # [(angle, z), (angle, z)]
            if electrodes.monopolar:
                pole_ends = ((np.radians(e.angle_deg), e.position_cm / 100),)
            if not all(0.0 <= z <= self.limb_length for _, z in pole_ends):
                raise ValueError(f"electrode {e.name} ({e.position_cm} cm) is off the forearm: positions must be "
                                 f"within 0-{self.limb_length * 100:.1f} cm of the elbow")
            poles += [[self.skin_radius(z) * np.cos(angle), self.skin_radius(z) * np.sin(angle), z]
                      for angle, z in pole_ends]
        return np.array(poles).reshape(len(electrodes), -1, 3)

    def under_fat(self, z: np.ndarray) -> np.ndarray:
        """Radius (m) left for muscle at distance z from the elbow."""
        return self.skin_radius(z) - self.fat

    def xy_scale(self, z: np.ndarray) -> np.ndarray:
        """The generic model's cross-section -> this subject's, at distance z from the elbow."""
        return self.under_fat(z) * self.sub.radial_fill / R_MODEL_REF

    def bellies(self, d: mujoco.MjData) -> np.ndarray:
        """(nu, TRACK_POINTS, 3) the model's bellies at this pose with the elbow at the atlas's reference angle (they
        carry the posture to the atlas points). The forearm frame rides on the ulna, so the elbow angle only swings
        the humeral origins of the muscles that cross it around that frame (at 72 deg BRD would leave the limb by
        6 cm); pronation, the wrist and the fingers move the bellies as they are."""
        t = self._track
        t.qpos[:] = d.qpos
        t.qpos[self.model.elbow_qpos] = self.anatomy.q_ref[self.model.elbow_qpos]
        mujoco.mj_kinematics(self.model.m, t)
        return self.model.belly_points(TRACK_POINTS, t)

    def atlas_positions(self, pts: AtlasPoints, belly: np.ndarray, sel: np.ndarray | slice = slice(None)) -> np.ndarray:
        """(N, 3) prepared atlas points (Anatomy.prepare) in this subject's forearm at the current pose."""
        return self.anatomy.positions(pts, belly, self.under_fat, self.xy_scale, sel)

    def muscle_points(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) points filling the atlas muscles at this pose, and the muscle of each."""
        if not hasattr(self, "_volume"):
            muscles = sorted(self.anatomy.points)
            self._volume = self.anatomy.prepare(np.repeat(muscles, [len(self.anatomy.points[m]) for m in muscles]),
                                                np.vstack([self.anatomy.points[m] for m in muscles]))
        return self._inside(self.atlas_positions(self._volume, self.bellies(d))), self._volume.muscle

    def muscle_surfaces(self, d: mujoco.MjData) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """(muscle, vertices (V, 3) in the forearm frame at this pose, faces) of every atlas muscle surface; the
        vertices move like the volume points (muscle_points)."""
        if not self.anatomy.meshes:
            return []
        if not hasattr(self, "_surface"):
            ms = sorted(self.anatomy.meshes)
            self._surface_split = np.cumsum([len(self.anatomy.meshes[m][0]) for m in ms])[:-1]
            self._surface = self.anatomy.prepare(np.repeat(ms, [len(self.anatomy.meshes[m][0]) for m in ms]),
                                                 np.vstack([self.anatomy.meshes[m][0] for m in ms]))
        verts = np.split(self._inside(self.atlas_positions(self._surface, self.bellies(d))), self._surface_split)
        return [(m, v, self.anatomy.meshes[m][1]) for m, v in zip(sorted(self.anatomy.meshes), verts)]

    def _inside(self, p: np.ndarray) -> np.ndarray:
        """Points pulled back under the fat where posture pushed them out (as the units are)."""
        r, under_fat = np.hypot(p[:, 0], p[:, 1]), self.under_fat(p[:, 2])
        p[:, :2] *= (np.minimum(r, under_fat) / np.maximum(r, 1e-9))[:, None]
        return p

    def bone_surfaces(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """(name, vertices in the forearm frame, faces) of the atlas radius and ulna, in this subject's section at
        the atlas's reference pose (they do not follow posture)."""
        out = []
        for name, (v, f) in self.anatomy.bone_meshes.items():
            z = self.anatomy.z(v[:, 0])
            r = self.under_fat(z) * np.minimum(v[:, 2], 1.0)
            out.append((name, np.column_stack([r * np.cos(v[:, 1]), r * np.sin(v[:, 1]), z]), f))
        return out

    def _bipolar(self, pole_emg: np.ndarray) -> np.ndarray:
        """(N, C) pole 1 minus pole 2 delayed by the conduction time between them."""
        pole1, pole2 = pole_emg[:, 0::2], pole_emg[:, 1::2]
        t = np.arange(len(pole_emg))
        return np.stack([pole1[:, c] - np.interp(t - self.pole_delay[c] * self.fs, t, pole2[:, c], left=0.0)
                         for c in range(len(self.poles))], axis=1)
