"""Plots for the BioMime backend.

    python -m src.biomime.inspect trial <shard.h5> [trial] --out bm_trial.png   # spikes raster + EMG + drive
    python -m src.biomime.inspect muaps [--muscle ECRB_r] --out muaps.png        # bank MUAPs on the 10x32 grid
"""
import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bank import BANK_PATH, MuapBank


def plot_trial(shard, trial, out, seconds=10.0):
    with h5py.File(shard, "r") as f:
        name = trial or sorted(f.keys())[0]
        g = f[name]
        if "spikes" not in g:
            raise SystemExit(f"{name} has no spikes: generate with --emg-model biomime --save-spikes")
        fs_emg, fs_l = g.attrs["fs_emg"], g.attrs["fs_label"]
        emg = g["emg"][: int(seconds * fs_emg)]
        unit, sample = g["spikes/unit"][:], g["spikes/sample"][:]
        muscle = g["spikes/units/muscle"][:]
        rank = g["spikes/units/rank"][:]
        names = list(g["spikes/units"].attrs["muscles"])
        exc = g["labels/exc"][: int(seconds * fs_l)]
    keep = sample < seconds * 2048
    unit, sample = unit[keep], sample[keep]
    order = np.lexsort((rank, muscle))                       # rows grouped by muscle, then by size
    row_of = np.empty_like(order)
    row_of[order] = np.arange(len(order))
    fig, ax = plt.subplots(3, 1, figsize=(13, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1]})
    ax[0].scatter(sample / 2048, row_of[unit], s=0.3, c=muscle[unit] % 10, cmap="tab10")
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
    print(f"wrote {out}")


def plot_muaps(muscle, out, path=BANK_PATH):
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
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["trial", "muaps"])
    ap.add_argument("shard", nargs="?")
    ap.add_argument("trial", nargs="?")
    ap.add_argument("--muscle", default=None)
    ap.add_argument("--out", default="biomime.png")
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    if a.cmd == "trial":
        plot_trial(a.shard, a.trial, a.out, a.seconds)
    else:
        plot_muaps(a.muscle, a.out)


if __name__ == "__main__":
    main()
