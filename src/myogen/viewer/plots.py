"""PNG plots of generated data: `myogen inspect`, `inspect-biomime` and `inspect-muaps`."""
import json
import logging

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..body.model import Anatomy, load
from ..motion.latent import LATENT_NAMES
from ..simulation.trial import FS_BIOMIME

log = logging.getLogger(__name__)


def plot_trial(shard, trial=None, out="trial.png", seconds=10.0):
    """`myogen inspect`: EMG, envelopes, latent, joint angles and excitations of one trial."""
    m, d = load()
    anat = Anatomy.from_model(m, d)
    with h5py.File(shard, "r") as f:
        name = trial or sorted(f.keys())[0]
        g = f[name]
        fe, fl = g.attrs["fs_emg"], g.attrs["fs_label"]
        emg = g["emg"][: int(seconds * fe)]
        lab = {k: g[f"labels/{k}"][: int(seconds * fl)] for k in g["labels"]}
        qc = json.loads(g.attrs["qc"])
        amputee = bool(g.attrs["amputee"])
    te, tl = np.arange(len(emg)) / fe, np.arange(len(lab["q"])) / fl
    C = emg.shape[1]
    win = max(1, int(0.1 * fe))
    env = np.sqrt(np.stack([np.convolve(emg[:, c] ** 2, np.ones(win) / win, "same") for c in range(C)], 1))

    fig, ax = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    off = 4 * np.percentile(np.abs(emg), 99)
    for c in range(C):
        ax[0].plot(te, emg[:, c] + c * off, lw=0.3)
    ax[0].set_title(f"{name}  amputee={amputee}  rmse={qc['rmse_deg']:.2f} deg  pass={qc['pass']}  -- raw sEMG (mV, offset)")
    ax[1].plot(te, env, lw=0.8)
    ax[1].set_title("RMS envelopes (100 ms)")
    for i, n in enumerate(LATENT_NAMES):
        ax[2].plot(tl, lab["latent"][:, i], label=n, lw=1)
    ax[2].legend(ncol=5, fontsize=7)
    ax[2].set_title("intent latent")
    for name in ["elbow_flexion_r", "pro_sup_r", "flexion_r", "cmc_flexion_r", "mcp2_flexion_r", "pm2_flexion_r"]:
        if name not in anat.joint_names:
            continue
        j = anat.joint_names.index(name)
        line, = ax[3].plot(tl, np.degrees(lab["q"][:, j]), label=name)
        ax[3].plot(tl, np.degrees(lab["q_ref"][:, j]), "--", color=line.get_color(), lw=0.8)
    ax[3].legend(ncol=6, fontsize=7)
    ax[3].set_title("joint angles (solid: simulated, dashed: reference), deg")
    top = np.argsort(-lab["exc"].mean(0))[:10]
    for i in top:
        ax[4].plot(tl, lab["exc"][:, i], label=anat.muscle_names[i], lw=0.8)
    ax[4].plot(tl, lab["cocontraction"], "k--", label="co-contraction", lw=1)
    ax[4].plot(tl, lab["grip_load"], "k:", label="grip load", lw=1)
    ax[4].legend(ncol=6, fontsize=7)
    ax[4].set_title("neural excitation (top 10 muscles)")
    ax[4].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    log.info(f"wrote {out}")


def plot_spikes(shard, trial=None, out="trial.png", seconds=10.0):
    """`myogen inspect-biomime`: spike raster, EMG and drive of a BioMime trial generated with save_spikes."""
    with h5py.File(shard, "r") as f:
        name = trial or sorted(f.keys())[0]
        g = f[name]
        if "spikes" not in g:
            raise ValueError(f"{name} has no spikes: generate with generate.emg_model=biomime generate.save_spikes=true")
        fs_emg, fs_l = g.attrs["fs_emg"], g.attrs["fs_label"]
        emg = g["emg"][: int(seconds * fs_emg)]
        unit, sample = g["spikes/unit"][:], g["spikes/sample"][:]
        muscle = g["spikes/units/muscle"][:]
        rank = g["spikes/units/rank"][:]
        names = list(g["spikes/units"].attrs["muscles"])
        exc = g["labels/exc"][: int(seconds * fs_l)]
    keep = sample < seconds * FS_BIOMIME
    unit, sample = unit[keep], sample[keep]
    order = np.lexsort((rank, muscle))                       # rows grouped by muscle, then by size
    row_of = np.empty_like(order)
    row_of[order] = np.arange(len(order))
    fig, ax = plt.subplots(3, 1, figsize=(13, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1]})
    ax[0].scatter(sample / FS_BIOMIME, row_of[unit], s=0.3, c=muscle[unit] % 10, cmap="tab10")
    bounds = np.flatnonzero(np.diff(muscle[order])) + 0.5
    for b in bounds:
        ax[0].axhline(b, color="0.85", lw=0.5)
    starts = np.r_[0, bounds + 0.5].astype(int)
    for s0, mi in zip(starts, muscle[order][starts]):
        ax[0].text(seconds * 1.002, s0, names[mi], fontsize=6, va="bottom")
    ax[0].set_ylabel("motor unit (by muscle, size)")
    ax[0].set_title(f"{name}: {len(np.unique(unit))} active units, {len(unit)} spikes")
    t = np.arange(len(emg)) / fs_emg
    off = 4 * np.percentile(np.abs(emg), 99) + 1e-9
    for c in range(emg.shape[1]):
        ax[1].plot(t, emg[:, c] + c * off, lw=0.3)
    ax[1].set_ylabel("EMG (device units, offset)")
    ax[2].plot(np.arange(len(exc)) / fs_l, exc, lw=0.6)
    ax[2].set_ylabel("neural drive")
    ax[2].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    log.info(f"wrote {out}")


def plot_muaps(muscle, out, path):
    """`myogen inspect-muaps`: bank MUAPs of one muscle on BioMime's 10 x 32 grid, across fibre-length bins."""
    from ..emg.biomime.bank import MuapBank
    bank = MuapBank(path)
    muscle = muscle or bank.muscles[0]
    table = bank.unit_table(muscle)
    pick = np.unique(np.linspace(0, len(table["size"]) - 1, 4).astype(int))
    muap = bank.muaps(muscle, pick)                          # (k, nL, 10, 32, 96)
    fig, ax = plt.subplots(len(pick), 2, figsize=(14, 3 * len(pick)))
    for r, (u, x) in enumerate(zip(pick, muap)):
        ref = x[2]                                           # L = 1
        rms = np.sqrt((ref ** 2).mean(-1))
        ax[r, 0].imshow(rms, aspect="auto", cmap="magma")
        ax[r, 0].set_title(f"{muscle} unit {u}: {table['size'][u]:.0f} fibres, RMS on the 10x32 grid (uV)")
        ax[r, 0].set_xlabel("column (angle)")
        ax[r, 0].set_ylabel("row (along fibres)")
        i, j = np.unravel_index(np.argmax(rms), rms.shape)
        for b, L in enumerate(bank.L_bins):
            ax[r, 1].plot(np.arange(96) / 2.048, x[b, i, j], label=f"L={L:.3f}")
        ax[r, 1].set_title(f"waveform at row {i}, col {j} vs fibre length")
        ax[r, 1].set_xlabel("ms")
    ax[0, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    log.info(f"wrote {out}")
