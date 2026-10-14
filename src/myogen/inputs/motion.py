from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..simulation.musculoskeletal import ELBOW_JOINT, HAND_JOINTS

MIN_DURATION = 0.5          # s
SUFFIXES = (".h5", ".hdf5", ".npz")

# Relaxed hand (rad): fingers slightly flexed, thumb resting, wrist neutral. Used for joints nobody measured.
REST_POSTURE = {
    "pro_sup_r": 0.0, "deviation_r": 0.0, "flexion_r": 0.0,
    "cmc_flexion_r": 0.075, "cmc_abduction_r": -0.015, "mp_flexion_r": 0.0, "ip_flexion_r": -0.1,
    "mcp2_flexion_r": 0.28, "mcp2_abduction_r": 0.0, "pm2_flexion_r": 0.3, "md2_flexion_r": 0.22,
    "mcp3_flexion_r": 0.35, "mcp3_abduction_r": 0.0, "pm3_flexion_r": 0.375, "md3_flexion_r": 0.27,
    "mcp4_flexion_r": 0.42, "mcp4_abduction_r": 0.0, "pm4_flexion_r": 0.45, "md4_flexion_r": 0.33,
    "mcp5_flexion_r": 0.42, "mcp5_abduction_r": 0.0, "pm5_flexion_r": 0.45, "md5_flexion_r": 0.33,
}
DIP_PER_PIP = 0.725         # DIP-PIP coupling of the fingers (the DIP cannot be moved on its own)

log = logging.getLogger(__name__)


@dataclass
class HandMotion:
    """The input movement: hand and wrist joint angles over time, at any sample rate (file format: README)."""
    angles: np.ndarray                   # (T, 23) radians, columns in musculoskeletal.HAND_JOINTS order
    fs: float                            # Hz
    elbow: np.ndarray | None = None      # (T,) radians; None = the subject's elbow angle
    rest: np.ndarray | None = None       # (T,) bool, True where the hand is relaxed
    filled: tuple[str, ...] = ()         # joints that were not given (filled by JointAdapter)

    def __post_init__(self) -> None:
        self.angles = np.asarray(self.angles, float)
        self.fs = float(self.fs) if np.ndim(self.fs) == 0 else self.fs
        if self.angles.ndim != 2 or self.angles.shape[1] != len(HAND_JOINTS):
            raise ValueError(f"angles must have shape (T, {len(HAND_JOINTS)}), got {self.angles.shape}")
        if not isinstance(self.fs, float) or not self.fs > 0:
            raise ValueError(f"fs must be a positive sample rate in Hz, got {self.fs!r}")
        if not np.isfinite(self.angles).all():
            raise ValueError("angles contain NaN / inf values")
        if self.duration < MIN_DURATION:
            raise ValueError(f"motion is {self.duration:.2f} s long; at least {MIN_DURATION} s is needed")
        for name in ("elbow", "rest"):
            v = getattr(self, name)
            if v is not None:
                v = np.asarray(v, bool if name == "rest" else float).ravel()
                if len(v) != len(self.angles):
                    raise ValueError(f"{name} has {len(v)} samples, angles have {len(self.angles)}")
                setattr(self, name, v)

    @property
    def duration(self) -> float:
        return len(self.angles) / self.fs

    def resampled(self, fs: float) -> HandMotion:
        """A copy at fs: angles linearly interpolated, rest labels by nearest sample."""
        t_in = np.arange(len(self.angles)) / self.fs
        t = np.arange(int(round(self.duration * fs))) / fs
        interp = lambda x: np.stack([np.interp(t, t_in, c) for c in np.atleast_2d(x.T)], axis=1)
        nearest = np.minimum(np.round(t * self.fs).astype(int), len(self.angles) - 1)
        return HandMotion(interp(self.angles), fs, None if self.elbow is None else interp(self.elbow)[:, 0],
                          None if self.rest is None else self.rest[nearest], self.filled)

    @classmethod
    def load(cls, path: str | Path, adapter: JointAdapter | None = None) -> HandMotion:
        """Read a motion file (format: module docstring); `adapter` maps joint names other than the model's."""
        path = _checked(path)
        if not path.exists():
            raise FileNotFoundError(f"motion file {path} not found")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as f:
                data = {k: f[k] for k in f.files}
        else:
            import h5py
            with h5py.File(path, "r") as f:
                data = {k: f[k][()] for k in ("angles", "joint_names", "rest", "fs") if k in f}
                data.update({k: v for k, v in f.attrs.items() if k in ("fs", "joint_names")})
        for key in ("angles", "joint_names", "fs"):
            if key not in data:
                raise ValueError(f"{path}: no '{key}' (see the motion file format in myogen.motion)")
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in np.asarray(data["joint_names"]).ravel()]
        angles = np.asarray(data["angles"], float)
        if angles.ndim != 2 or angles.shape[1] != len(names):
            raise ValueError(f"{path}: angles {angles.shape} do not match {len(names)} joint_names")
        return (adapter or JointAdapter())(dict(zip(names, angles.T)), float(np.asarray(data["fs"])),
                                           rest=data.get("rest"))

    def save(self, path: str | Path) -> Path:
        path = _checked(path)
        names = list(HAND_JOINTS) + ([ELBOW_JOINT] if self.elbow is not None else [])
        angles = self.angles if self.elbow is None else np.column_stack([self.angles, self.elbow])
        extra = {} if self.rest is None else {"rest": self.rest}
        if path.suffix == ".npz":
            np.savez(path, angles=angles, joint_names=np.array(names), fs=self.fs, **extra)
        else:
            import h5py
            with h5py.File(path, "w") as f:
                f.create_dataset("angles", data=angles)
                for k, v in extra.items():
                    f.create_dataset(k, data=v)
                f.attrs["fs"], f.attrs["joint_names"] = self.fs, names
        return path


class JointAdapter:
    """Your joint angles -> HandMotion in the model's joints.

    mapping: {your_name: model_joint} or {your_name: (model_joint, sign, offset_rad)}, for names that differ from
    the model's (musculoskeletal.HAND_JOINTS, elbow_flexion_r); model names pass through without a mapping. Joints
    you do not measure are filled: a finger's MCP and PIP from each other, its DIP from the PIP (DIP_PER_PIP),
    anything else at the relaxed posture (REST_POSTURE).
    """

    def __init__(self, mapping: Mapping[str, str | tuple[str, float, float]] | None = None,
                 degrees: bool = False) -> None:
        self.mapping: dict[str, tuple[str, float, float]] = {
            k: (v, 1.0, 0.0) if isinstance(v, str) else tuple(v) for k, v in (mapping or {}).items()}
        self.degrees = degrees
        known = set(HAND_JOINTS) | {ELBOW_JOINT}
        bad = sorted(v[0] for v in self.mapping.values() if v[0] not in known)
        if bad:
            raise ValueError(f"unknown model joints {bad}; model joints are {HAND_JOINTS + [ELBOW_JOINT]}")

    def __call__(self, angles: Mapping[str, np.ndarray], fs: float, rest: np.ndarray | None = None) -> HandMotion:
        """angles: {name: (T,) array}. Returns a HandMotion."""
        given = {}
        for name, values in angles.items():
            joint, sign, offset = self.mapping.get(name, (name, 1.0, 0.0))
            if joint not in REST_POSTURE and joint != ELBOW_JOINT:
                raise ValueError(f"joint {name!r} is not a model joint and has no mapping (JointAdapter(mapping))")
            values = np.asarray(values, float)
            given[joint] = sign * (np.radians(values) if self.degrees else values) + offset
        if not given:
            raise ValueError("no joint angles given")
        T = len(next(iter(given.values())))
        q, filled = _fill(given, T)
        if filled:
            log.info(f"filled {len(filled)} joints not given: {', '.join(filled)}")
        return HandMotion(np.column_stack([q[j] for j in HAND_JOINTS]), fs, given.get(ELBOW_JOINT), rest,
                          tuple(filled))


def _fill(given: dict[str, np.ndarray], T: int) -> tuple[dict[str, np.ndarray], list[str]]:
    """All hand joints from the given ones (see JointAdapter). Returns ({joint: (T,)}, filled names)."""
    q = {j: v for j, v in given.items() if j != ELBOW_JOINT}
    for f in "2345":
        mcp, pip, dip = f"mcp{f}_flexion_r", f"pm{f}_flexion_r", f"md{f}_flexion_r"
        if pip not in q and mcp in q:
            q[pip] = q[mcp]
        if mcp not in q and pip in q:
            q[mcp] = q[pip]
        if dip not in q and pip in q:
            q[dip] = DIP_PER_PIP * q[pip]
    filled = [j for j in HAND_JOINTS if j not in given]
    for j in HAND_JOINTS:
        q.setdefault(j, np.full(T, REST_POSTURE[j]))
    return q, filled


def example_motion(duration: float = 6.0, fs: float = 100.0) -> HandMotion:
    """Open-close grasps with wrist flexion and pronation, for trying the tool (all joints given)."""
    t = np.arange(int(duration * fs)) / fs
    close = 0.5 - 0.5 * np.cos(2 * np.pi * t / 3.0)                  # 0 open .. 1 closed, every 3 s
    q = {j: np.full(len(t), v) for j, v in REST_POSTURE.items()}
    for f in "2345":
        q[f"mcp{f}_flexion_r"] = q[f"pm{f}_flexion_r"] = 0.1 + 1.2 * close
        q[f"md{f}_flexion_r"] = DIP_PER_PIP * q[f"pm{f}_flexion_r"]
    q["cmc_flexion_r"], q["ip_flexion_r"] = 0.2 - 0.6 * close, 0.1 - 0.8 * close
    q["flexion_r"] = 0.4 * np.sin(2 * np.pi * t / 6.0)
    q["pro_sup_r"] = 0.5 * np.sin(2 * np.pi * t / 4.0)
    return HandMotion(np.column_stack([q[j] for j in HAND_JOINTS]), fs, rest=close < 0.05)


def _checked(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix not in SUFFIXES:
        raise ValueError(f"motion file must be .h5, .hdf5 or .npz, got {path.name}")
    return path
