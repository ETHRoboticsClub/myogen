"""The simulation input: a movement recording, i.e. a trajectory of the posture latent (see latent.py).

File format (.h5 / .hdf5, or .npz with the same names):

    latent        float (T, 10)   required, columns in LATENT_NAMES order
    fs            float           required (HDF5: file attribute), sample rate in Hz (any rate works)
    restimulus    int (T,)        optional, 0 where the hand is at rest (sets the subject's relaxed posture)
    latent_names  str (10,)       optional (HDF5: file attribute), checked against LATENT_NAMES
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .latent import LATENT_HI, LATENT_LO, LATENT_NAMES, N_LATENT
from .synthetic import intent_trajectory

RANGE_TOLERANCE = 0.05          # values this far outside the latent range are clipped with a warning; further: error
MIN_DURATION = 0.5              # s
SUFFIXES = (".h5", ".hdf5", ".npz")

log = logging.getLogger(__name__)


@dataclass
class Motion:
    latent: np.ndarray                      # (T, 10)
    fs: float                               # Hz
    restimulus: np.ndarray | None = None    # (T,) int, 0 = rest

    def __post_init__(self):
        self.latent = np.array(self.latent, dtype=float)
        if self.restimulus is not None:
            self.restimulus = np.asarray(self.restimulus).astype(np.int16).ravel()
        self.validate()

    @property
    def duration(self):
        return len(self.latent) / self.fs

    def validate(self):
        """Raise ValueError on a malformed recording; clip values marginally outside the latent range."""
        z = self.latent
        if z.ndim != 2 or z.shape[1] != N_LATENT:
            raise ValueError(f"latent must have shape (T, {N_LATENT}) with columns {LATENT_NAMES}, got {z.shape}")
        if np.ndim(self.fs) != 0 or not self.fs > 0:
            raise ValueError(f"fs must be a positive sample rate in Hz, got {self.fs!r}")
        self.fs = float(self.fs)
        if not np.isfinite(z).all():
            rows = np.flatnonzero(~np.isfinite(z).all(1))
            raise ValueError(f"latent has NaN / inf values ({len(rows)} samples, first at index {rows[0]})")
        if self.duration < MIN_DURATION:
            raise ValueError(f"motion is {self.duration:.2f} s long; at least {MIN_DURATION} s is needed")
        if self.restimulus is not None and len(self.restimulus) != len(z):
            raise ValueError(f"restimulus has {len(self.restimulus)} samples, latent has {len(z)}")
        excess = np.maximum(LATENT_LO - z, z - LATENT_HI).max(0)
        bad = [f"{LATENT_NAMES[i]} (by {excess[i]:.2f})" for i in np.flatnonzero(excess > RANGE_TOLERANCE)]
        if bad:
            raise ValueError(f"latent out of range in {', '.join(bad)}: closures, thumb_abd and spread are in "
                             f"[0, 1], wrist and pro_sup in [-1, 1]")
        if (excess > 0).any():
            log.warning(f"clipping latent values slightly out of range: "
                        f"{[LATENT_NAMES[i] for i in np.flatnonzero(excess > 0)]}")
            self.latent = np.clip(z, LATENT_LO, LATENT_HI)

    def resampled(self, fs):
        """(latent, restimulus) at fs: linear interpolation of the latent, nearest sample of restimulus."""
        t_in = np.arange(len(self.latent)) / self.fs
        t = np.arange(int(round(self.duration * fs))) / fs
        z = np.stack([np.interp(t, t_in, self.latent[:, i]) for i in range(N_LATENT)], axis=1)
        restim = None
        if self.restimulus is not None:
            restim = self.restimulus[np.minimum(np.round(t * self.fs).astype(int), len(self.restimulus) - 1)]
        return z, restim

    @classmethod
    def load(cls, path):
        path = _checked_suffix(path)
        if not path.exists():
            raise FileNotFoundError(f"motion file {path} not found")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as f:
                data = {k: f[k] for k in f.files}
            fs, names = data.get("fs"), data.get("latent_names")
        else:
            import h5py
            with h5py.File(path, "r") as f:
                data = {k: f[k][()] for k in ("latent", "restimulus") if k in f}
                fs = f.attrs.get("fs", f["fs"][()] if "fs" in f else None)
                names = f.attrs.get("latent_names")
        if "latent" not in data:
            raise ValueError(f"{path}: no 'latent' dataset (T, {N_LATENT})")
        if fs is None:
            raise ValueError(f"{path}: no 'fs' (sample rate in Hz; HDF5: file attribute)")
        if names is not None:
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in np.asarray(names).ravel()]
            if names != LATENT_NAMES:
                raise ValueError(f"{path}: latent_names {names} do not match the expected order {LATENT_NAMES}")
        return cls(latent=data["latent"], fs=float(np.asarray(fs)), restimulus=data.get("restimulus"))

    def save(self, path):
        path = _checked_suffix(path)
        if path.suffix == ".npz":
            extra = {} if self.restimulus is None else {"restimulus": self.restimulus}
            np.savez(path, latent=self.latent, fs=self.fs, latent_names=np.array(LATENT_NAMES), **extra)
        else:
            import h5py
            with h5py.File(path, "w") as f:
                f.create_dataset("latent", data=self.latent)
                if self.restimulus is not None:
                    f.create_dataset("restimulus", data=self.restimulus)
                f.attrs["fs"] = self.fs
                f.attrs["latent_names"] = LATENT_NAMES
        return path


def example_motion(duration=10.0, seed=0, fs=100.0):
    """A valid Motion from the synthetic movement generator (grasps, holds, returns to rest)."""
    rng = np.random.default_rng(seed)
    return Motion(latent=intent_trajectory(rng, duration, fs, speed=rng.uniform(0.7, 1.4)), fs=fs)


def _checked_suffix(path):
    path = Path(path)
    if path.suffix not in SUFFIXES:
        raise ValueError(f"motion file must be .h5, .hdf5 or .npz, got {path.name}")
    return path
