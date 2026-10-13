"""The pretrained BioMime generator: physical motor-unit conditions in, monopolar MUAPs (volts) out.

Conditions are normalised as in training ([min, max] -> [0.5, 1]); values outside the training range are clipped.
"""
from pathlib import Path

import numpy as np
import torch
import yaml

from ...config import user_data_dir

CHECKPOINT = user_data_dir() / "biomime" / "model_linear.pth"   # default; commands pass paths.biomime_weights
CONFIG = Path(__file__).resolve().parent / "vendor" / "generator.yaml"

# Training ranges of the public checkpoint (BioMime/utils/params.py), physical units.
RANGES = {
    "num": (1.098, 6.77),        # log(number of fibres)
    "depth": (0.007, 0.0231),    # m below the skin
    "angle": (0.0, 1.0),         # fraction of the circumference (grid column / 32)
    "iz": (0.365, 0.636),        # innervation zone, fraction of fibre length
    "cv": (3.0, 4.5),            # m/s
    "len": (0.85, 1.15),         # fibre length relative to the reference anatomy
}
COND_ORDER = ("num", "depth", "angle", "iz", "cv", "len")
W_AMP = 1e5          # BioMime trains on MUAPs x 1e5; outputs are divided by this to get volts
FS = 2048            # MUAP sampling rate (Hz)
N_TIME, N_ROW, N_COL = 96, 10, 32
LATENT = 16


def normalise(name, x):
    lo, hi = RANGES[name]
    return 0.5 + 0.5 * (np.asarray(x, np.float64) - lo) / (hi - lo)


def clip_physical(cond):
    """cond: dict name -> array. Returns clipped copy and a dict of boolean masks of clipped entries."""
    out, clipped = {}, {}
    for k in COND_ORDER:
        lo, hi = RANGES[k]
        v = np.asarray(cond[k], np.float64)
        out[k] = np.clip(v, lo, hi)
        clipped[k] = (v < lo) | (v > hi)
    return out, clipped


class _Attrs(dict):
    __getattr__ = dict.__getitem__


def _attrs(x):
    """Nested dicts with attribute access, as BioMime's Generator reads its config (cfg.Enc.num_layers)."""
    return _Attrs({k: _attrs(v) for k, v in x.items()}) if isinstance(x, dict) else x


class BioMimeGenerator:
    """Thin wrapper around BioMime's Generator: physical conditions in, MUAPs (volts) out."""

    def __init__(self, checkpoint=CHECKPOINT, config=CONFIG, device="cpu", threads=None):
        from .vendor.generator import Generator
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"BioMime weights {checkpoint} missing: `myogen download biomime-weights`")
        if threads:
            torch.set_num_threads(threads)
        with open(config) as f:
            self.net = Generator(_attrs(yaml.safe_load(f)))
        # weights_only: the checkpoint is a plain state dict, never unpickle arbitrary objects.
        self.net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        self.net.eval().to(device)
        self.device = device

    @torch.no_grad()
    def sample(self, cond, z, batch=256):
        """cond: dict of (n,) physical arrays (already clipped); z: (n, 16) latents.
        Returns (n, N_ROW, N_COL, N_TIME) float32 MUAPs in volts (monopolar)."""
        c = np.stack([normalise(k, cond[k]) for k in COND_ORDER], axis=1).astype(np.float32)
        n = len(c)
        out = np.empty((n, N_ROW, N_COL, N_TIME), np.float32)
        for i in range(0, n, batch):
            cb = torch.from_numpy(c[i:i + batch]).to(self.device)
            zb = torch.from_numpy(np.asarray(z[i:i + batch], np.float32)).to(self.device)
            y = self.net.sample(len(cb), cb, self.device, zb)          # (b, 96, 10, 32)
            out[i:i + batch] = y.permute(0, 2, 3, 1).cpu().numpy() / W_AMP
        return out
