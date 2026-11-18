from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from ..inputs.subject import Subject
    from .musculoskeletal import ForearmModel

ANATOMY_NPZ = Path(__file__).resolve().parent.parent / "models" / "forearm_anatomy.npz"
TRACK_POINTS = 12          # belly points that carry the posture to the atlas points
MIN_DEPTH_FRACTION = 0.05  # no muscle point nearer the axis than this (the bones)


class Anatomy:
    """Where each forearm muscle is, from BodyParts3D (models/build_anatomy.py, docs/anatomy.md): points sampled
    uniformly over the muscle's volume as (level along the radius, angle, depth fraction), at the model's reference
    pose. Posture moves every point with its muscle's belly in the model (the belly point at the same height), so the
    cross-section comes from the atlas and the motion from the musculoskeletal model. Muscles outside the atlas (upper
    arm) have no points and keep the lines of action."""

    def __init__(self, model: ForearmModel, subject: Subject | None = None, path: str | Path = ANATOMY_NPZ) -> None:
        self.model = model
        data = np.load(path)
        names = [str(n) for n in data["names"]]
        self.has = np.array([n in names for n in model.muscle_names])
        self.points = {i: data[f"points_{n}"].astype(float) for i, n in enumerate(model.muscle_names) if n in names}
        if subject is not None:
            self.points = {i: self._vary(i, p, subject) for i, p in self.points.items()}
        d = mujoco.MjData(model.m)
        d.qpos[:] = data["q_ref"]
        mujoco.mj_kinematics(model.m, d)
        self.z_head, self.z_tip = model.radius_extent(d)
        self.belly_ref = model.belly_points(TRACK_POINTS, d)

    def _vary(self, muscle: int, points: np.ndarray, subject: Subject) -> np.ndarray:
        """This subject's version of a muscle's points: the section turned, the muscle shifted around the forearm,
        nearer or farther from the skin, and wider or narrower around its centre at each level."""
        n = self.model.nu
        size = subject.per_muscle("muscle_size", n, 1.0)[muscle]
        depth = subject.per_muscle("muscle_depth", n, 1.0)[muscle]
        turn = np.radians(subject.section_rotation_deg + subject.per_muscle("muscle_shift_deg", n, 0.0)[muscle])
        out = points.copy()
        for level in np.unique(points[:, 0]):
            sel = points[:, 0] == level
            angle, frac = points[sel, 1], points[sel, 2]
            centre = np.angle(np.mean(np.exp(1j * angle)))
            spread = np.sqrt(size)                                  # area ~ size
            out[sel, 1] = centre + spread * np.angle(np.exp(1j * (angle - centre)))
            out[sel, 2] = frac.mean() + spread * (frac - frac.mean())
        out[:, 1] = np.angle(np.exp(1j * (out[:, 1] + turn)))
        out[:, 2] = np.clip(1 - (1 - out[:, 2]) * depth, MIN_DEPTH_FRACTION, 1.0)   # depth below the muscle surface
        return out

    def z(self, level: np.ndarray) -> np.ndarray:
        """Distance from the elbow (m) of a level along the radius (0 head, 1 styloid) at the reference pose."""
        return self.z_head + level * (self.z_tip - self.z_head)

    def centroids(self, muscle: int, k: int) -> np.ndarray:
        """(k, 3) points standing for the muscle in k slabs of equal volume, proximal to distal: the noise engine's
        sources."""
        p = self.points[muscle]
        groups = np.array_split(p[np.argsort(p[:, 0], kind="stable")], k)
        return np.array([[g[:, 0].mean(), np.angle(np.mean(np.exp(1j * g[:, 1]))), g[:, 2].mean()] for g in groups])

    def prepare(self, muscles: np.ndarray, points: np.ndarray) -> AtlasPoints:
        """Fixed part of placing atlas points (N, 3: level, angle, depth) of these muscles (N,)."""
        z0 = self.z(points[:, 0])
        # Each point follows the reference belly point nearest in height.
        track = np.argmin(np.abs(z0[:, None] - self.belly_ref[muscles, :, 2]), axis=1)
        depth = np.minimum(points[:, 2], 1.0)
        return AtlasPoints(muscles, track, z0, depth * np.cos(points[:, 1]), depth * np.sin(points[:, 1]),
                           self.belly_ref[muscles, track])

    @staticmethod
    def positions(pts: AtlasPoints, belly: np.ndarray, under_fat, xy_scale,
                  sel: np.ndarray | slice = slice(None)) -> np.ndarray:
        """(N, 3) positions in the forearm frame of prepared points (or the subset sel) at the current pose.
        belly: (nu, TRACK_POINTS, 3) the model's bellies now; under_fat(z): radius left for muscle; xy_scale(z): model
        cross-section -> this subject's (for the belly displacement)."""
        shift = belly[pts.muscle[sel], pts.track[sel]] - pts.ref[sel]
        z = pts.z0[sel] + shift[:, 2]
        r, k = under_fat(z), xy_scale(z)
        return np.column_stack([r * pts.cos[sel] + k * shift[:, 0], r * pts.sin[sel] + k * shift[:, 1], z])


@dataclass(frozen=True)
class AtlasPoints:
    """Atlas points ready to place: muscle, tracked belly point, height, depth-scaled direction, reference belly."""
    muscle: np.ndarray
    track: np.ndarray
    z0: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    ref: np.ndarray
