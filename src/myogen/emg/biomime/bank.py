"""Offline MUAP bank: BioMime MUAPs of every visible muscle, motor units x fibre-length bins, full grid.

    myogen biomime-bank build | info | regrid

BioMime costs ~28 ms per MUAP, and NeuroMotion's movement dynamics depend on one scalar per muscle (normalised
fibre length L), so each unit is rendered once per L bin and synthesis interpolates between bins.

HDF5: attrs L_bins, grid, units ("uV"); per muscle: cond (n, 6), z (n, 16), size (n,), pool_index (n,),
amp_extra (n,), muap (n, nL, 10, 32, 96) float16 uV, and the territory as attributes.
"""
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from ...body.model import Anatomy, load
from ...config import user_data_dir
from .anatomy import GridGeometry, build_territories, mu_sizes
from .generator import COND_ORDER, LATENT, BioMimeGenerator, clip_physical

log = logging.getLogger(__name__)

BANK_PATH = user_data_dir() / "biomime" / "muap_bank.h5"   # default; commands pass paths.muap_bank
L_BINS = np.array([0.85, 0.925, 1.0, 1.075, 1.15])


def sample_units(rng, terr, n):
    """Physical BioMime conditions (at L = 1) for n bank units spanning the muscle's size distribution."""
    sizes_all = mu_sizes(terr.n_mu, terr.n_fibres)
    idx = np.unique(np.round(np.linspace(0, terr.n_mu - 1, n)).astype(int))
    sizes = sizes_all[idx]
    rank = idx / max(terr.n_mu - 1, 1)                                  # 0 = smallest, 1 = largest
    cond = {
        "num": np.log(np.maximum(sizes, 3.0)),     # clipped to BioMime's range later; excess -> amp_extra
        "depth": rng.uniform(*terr.depth, len(idx)),
        "angle": rng.uniform(*terr.angle, len(idx)) % 1.0,
        "iz": np.clip(rng.normal(0.5, 0.05, len(idx)), 0.37, 0.63),
        "cv": np.clip(3.3 + 1.0 * rank + rng.normal(0, 0.15, len(idx)), 3.0, 4.5),   # bigger -> faster
        "len": np.clip(rng.normal(1.0, 0.03, len(idx)), 0.9, 1.1),
    }
    return cond, sizes, idx


def dynamic_conditions(cond, L):
    """NeuroMotion's movement model: fibre length L scales depth (1/sqrt L), velocity (1/L) and length (L)."""
    out = dict(cond)
    out["depth"] = cond["depth"] / np.sqrt(L)
    out["cv"] = cond["cv"] / L
    out["len"] = cond["len"] * L
    return clip_physical(out)[0]


def build(out=BANK_PATH, per_muscle=128, seed=0, threads=None, grid=None, weights=None):
    grid = grid or GridGeometry()
    m, d = load()
    anat = Anatomy.from_model(m, d)
    terr = build_territories(m, d, anat, grid)
    gen = BioMimeGenerator(threads=threads) if weights is None else BioMimeGenerator(weights, threads=threads)
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    with h5py.File(out, "w") as f:
        f.attrs.update({"L_bins": L_BINS, "units": "uV", "grid": json.dumps(asdict(grid)),
                        "checkpoint": "model_linear.pth", "cond_order": list(COND_ORDER)})
        for name, t in terr.items():
            t0 = time.time()
            cond, sizes, idx = sample_units(rng, t, per_muscle)
            cond, _ = clip_physical(cond)
            # Units larger than BioMime's training range (870 fibres): same shape, amplitude x fibre ratio
            # (surface potentials superpose linearly over fibres).
            amp_extra = np.maximum(sizes / np.exp(cond["num"]), 1.0)
            n = len(sizes)
            z = rng.standard_normal((n, LATENT)).astype(np.float32)
            muap = np.empty((n, len(L_BINS), 10, 32, 96), np.float16)
            for b, L in enumerate(L_BINS):
                muap[:, b] = gen.sample(dynamic_conditions(cond, L), z) * 1e6
            g = f.create_group(name)
            g.create_dataset("cond", data=np.stack([cond[k] for k in COND_ORDER], 1).astype(np.float32))
            g.create_dataset("z", data=z)
            g.create_dataset("size", data=sizes.astype(np.float32))
            g.create_dataset("pool_index", data=idx.astype(np.int32))
            g.create_dataset("amp_extra", data=amp_extra.astype(np.float32))
            g.create_dataset("muap", data=muap, chunks=(1, len(L_BINS), 10, 32, 96))
            g.attrs.update({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(t).items()})
            g.attrs["gain"] = t.gain
            f.flush()
            log.info(f"{name:9s} {n:3d} units x {len(L_BINS)} bins  peak {np.abs(muap).max():7.1f} uV  "
                     f"({time.time() - t0:.0f} s)")
    log.info(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB) in {(time.time() - t_all) / 60:.1f} min")


class MuapBank:
    """Read access to the bank; muscles are loaded lazily and kept in memory."""

    def __init__(self, path=BANK_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"MUAP bank {path} missing: `myogen download muap-bank`, or build it with "
                                    f"`myogen biomime-bank build` (needs the BioMime weights)")
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.L_bins = f.attrs["L_bins"][:]
            self.grid = GridGeometry(**json.loads(f.attrs["grid"]))
            self.muscles = list(f.keys())
            self.meta = {k: dict(f[k].attrs) for k in self.muscles}

    def unit_table(self, muscle):
        with h5py.File(self.path, "r") as f:
            g = f[muscle]
            return {"cond": g["cond"][:], "z": g["z"][:], "size": g["size"][:], "pool_index": g["pool_index"][:],
                    "amp_extra": g["amp_extra"][:] if "amp_extra" in g else np.ones(len(g["size"]), np.float32)}

    def muaps(self, muscle, units=None):
        """(n, nL, 10, 32, 96) float32 microvolts for the selected units (all if None).

        Read from disk on every call (no per-process copy): parallel workers share the OS page cache instead
        of each holding the ~1 GB bank in memory.
        """
        with h5py.File(self.path, "r") as f:
            x = f[muscle]["muap"][:]          # contiguous read (h5py point selection is ~30x slower)
        x = x if units is None else x[np.sort(np.asarray(units))]
        return x.astype(np.float32)


def regrid(path=BANK_PATH, z0=None, z1=None):
    """Recompute territory metadata (axial overlap -> gain) for a new GridGeometry; MUAPs are unchanged
    because BioMime's output grid is fixed, only its placement on our forearm is an assumption."""
    grid = GridGeometry()
    if z0 is not None:
        grid.z0_frac = z0
    if z1 is not None:
        grid.z1_frac = z1
    m, d = load()
    terr = build_territories(m, d, Anatomy.from_model(m, d), grid)
    with h5py.File(path, "a") as f:
        f.attrs["grid"] = json.dumps(asdict(grid))
        for name in f:
            t = terr.get(name)
            gain = t.gain if t else 0.0
            f[name].attrs["axial_overlap"] = t.axial_overlap if t else 0.0
            f[name].attrs["gain"] = gain
            log.info(f"{name:9s} overlap {f[name].attrs['axial_overlap']:.2f} gain {gain:.3f}")
    log.info(f"regridded {path}: rows span {grid.z0_frac:.2f}-{grid.z1_frac:.2f} of forearm length")


def info(path=BANK_PATH):
    b = MuapBank(path)
    from rich.table import Table

    from ...log import console
    t = Table(title=f"{b.path}\nL bins {b.L_bins}, {b.grid}", title_justify="left")
    for col in ("muscle", "MUs", "gain", "depth (mm)", "angle (frac.)", "anchored"):
        t.add_column(col, justify="left" if col == "muscle" else "right")
    for k in b.muscles:
        mt = b.meta[k]
        t.add_row(k, str(mt["n_mu"]), f"{mt['gain']:.2f}", "-".join(f"{x * 1e3:.1f}" for x in mt["depth"]),
                  "-".join(f"{x:.3f}" for x in mt["angle"]), "yes" if mt["anchored"] else "")
    console.print(t)
