"""Plot one generated trial (EMG, envelopes, kinematics, excitations) to a PNG for sanity checking.

    python -m src.inspect_trial data/pilot/shard_000.h5 [trial] [--out trial.png] [--seconds 10]
"""
import argparse
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import Anatomy, load
from .motion import LATENT_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("trial", nargs="?")
    ap.add_argument("--out", default="trial.png")
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    m, d = load()
    anat = Anatomy.from_model(m, d)
    with h5py.File(a.shard, "r") as f:
        name = a.trial or sorted(f.keys())[0]
        g = f[name]
        fe, fl = g.attrs["fs_emg"], g.attrs["fs_label"]
        emg = g["emg"][: int(a.seconds * fe)]
        lab = {k: g[f"labels/{k}"][: int(a.seconds * fl)] for k in g["labels"]}
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
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
