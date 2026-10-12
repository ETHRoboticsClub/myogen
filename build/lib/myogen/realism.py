"""Realism gates: compare synthetic data with NinaPro DB1 / DB5 on the same statistics.

    python -m src.generate --out data/val_db1 --device otto_bock --motion ninapro --bank-datasets db1 ...
    python -m src.realism --real db1 --sim data/val_db1 --out reports/realism_db1

Both sides go through the same pipeline: 100 Hz envelopes (DB1: the electrodes' own output; DB5/Myo: 150 ms
moving RMS of the raw 200 Hz signal), normalised per subject and channel by the 99th percentile (as in the
emg2action training pipeline), then:

  amplitude    distribution of normalised envelopes; rest/movement ratio (NinaPro restimulus labels)
  correlation  envelope correlation between ring electrodes vs. their distance around the ring
  dimension    participation ratio of the ring envelopes (how many independent sources they expose)
  smoothness   envelope autocorrelation at 50 / 200 ms
  lag          rising EMG activity vs. hand speed cross-correlation peak (> 0: EMG leads)
  decoding     within-subject ridge regression log-envelope -> hand latent, R^2 on interleaved held-out blocks
  raw (Myo)    PSD shape, median frequency, |x| <= 1 count at rest, clipping

A gate passes when the synthetic value falls inside the real between-subject range (min..max, with a margin).
"""
import json
import logging
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .log import console
from .ninapro import DATA_DIR, list_subjects, load_mat, subject_files

log = logging.getLogger(__name__)

FS = 100
HAND = slice(0, 7)


def moving_rms(x, fs, win_s=0.15):
    w = max(1, int(win_s * fs))
    k = np.ones(w) / w
    return np.sqrt(np.maximum(signal.fftconvolve(x ** 2, k[:, None], mode="same", axes=0), 0))


def raw_to_env(raw, fs):
    env = moving_rms(raw, fs)
    return env[:: fs // FS] if fs > FS else env


# ---------------------------------------------------------------------------------------------- loading


def load_real(dataset, max_subjects=None, bank=DATA_DIR / "latent_bank.h5", root=DATA_DIR):
    """One record per subject: env (T, C), raw or None, restimulus (T,), latent (T, 10)."""
    subjects = list_subjects(dataset, root)[:max_subjects]
    if not subjects:
        raise FileNotFoundError(f"no NinaPro {dataset} subjects in {root / f'ninapro_{dataset}'} (paths.ninapro)")
    out = []
    with h5py.File(bank, "r") as fb:
        for s in subjects:
            env, raw, restim, lat = [], [], [], []
            for p in subject_files(dataset, s, root):
                r = load_mat(p)
                g = fb[f"{dataset}/s{s:02d}/e{r['exercise']}"]
                if dataset == "db5":
                    raw.append(r["emg"])
                    e = raw_to_env(r["emg"], r["fs"])
                else:
                    e = r["emg"]
                n = min(len(e), len(g["latent"]))
                env.append(e[:n]), restim.append(g["restimulus"][:n]), lat.append(g["latent"][:n])
            out.append({"subject": f"{dataset}-s{s}", "env": np.concatenate(env), "restim": np.concatenate(restim),
                        "latent": np.concatenate(lat), "raw": np.concatenate(raw) if raw else None,
                        "fs_raw": 200 if dataset == "db5" else None})
    return out


def load_sim(sim_dir, max_subjects=None):
    """Group generated trials by synthetic subject."""
    by_sub = {}
    for shard in sorted(Path(sim_dir).glob("shard_*.h5")):
        with h5py.File(shard, "r") as f:
            for name in f:
                g = f[name]
                if not json.loads(g.attrs["qc"])["pass"]:
                    continue
                dev = json.loads(g.attrs["subject_params"])["layout"]["device"]
                fs = int(g.attrs["fs_emg"])
                x = g["emg"][:]
                env = x if dev == "otto_bock" else raw_to_env(x, fs)
                lab = g["labels"]
                n = min(len(env), len(lab["latent"]))
                rec = by_sub.setdefault(int(g.attrs["subject"]), {"env": [], "raw": [], "restim": [], "latent": [],
                                                                    "fs_raw": fs if dev != "otto_bock" else None})
                rec["env"].append(env[:n]), rec["restim"].append(lab["restimulus"][:n])
                rec["latent"].append(lab["latent"][:n])
                if dev != "otto_bock":
                    rec["raw"].append(x)
    out = []
    for s, r in sorted(by_sub.items())[:max_subjects]:
        out.append({"subject": f"sim-s{s}", "env": np.concatenate(r["env"]), "restim": np.concatenate(r["restim"]),
                    "latent": np.concatenate(r["latent"]), "raw": np.concatenate(r["raw"]) if r["raw"] else None,
                    "fs_raw": r["fs_raw"]})
    return out


# ---------------------------------------------------------------------------------------------- statistics


def _ring(env):
    return env[:, :8]


def subject_stats(rec):
    env = rec["env"].astype(np.float64)
    env = env / (np.percentile(env, 99, axis=0) + 1e-9)
    restim, z = rec["restim"], rec["latent"]
    st = {}
    st["amp_quantiles"] = np.percentile(env, [5, 25, 50, 75, 95], axis=0).mean(1).tolist()
    rest, move = restim == 0, restim > 0
    if rest.any() and move.any():
        st["rest_move_ratio"] = float(np.mean(np.median(env[rest], 0) / (np.median(env[move], 0) + 1e-9)))
    ring = _ring(env)
    zc = (ring - ring.mean(0)) / (ring.std(0) + 1e-9)
    C = np.corrcoef(zc.T)
    nr = ring.shape[1]
    st["corr_vs_distance"] = [float(np.mean([C[i, (i + k) % nr] for i in range(nr)])) for k in range(nr // 2 + 1)]
    ev = np.linalg.eigvalsh(np.cov(zc.T))
    st["participation_ratio"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    zs = (env - env.mean(0)) / (env.std(0) + 1e-9)
    st["autocorr_50ms"] = float(np.mean([np.corrcoef(zs[:-5, c], zs[5:, c])[0, 1] for c in range(zs.shape[1])]))
    st["autocorr_200ms"] = float(np.mean([np.corrcoef(zs[:-20, c], zs[20:, c])[0, 1] for c in range(zs.shape[1])]))
    # Onset lag: rising EMG activity vs. hand speed (> 0: EMG leads). Holds would dominate a raw-level xcorr.
    act = signal.sosfiltfilt(signal.butter(2, 5, fs=FS, output="sos"), env.sum(1))
    rise = np.maximum(np.gradient(act), 0)
    speed = np.linalg.norm(np.gradient(z[:, HAND], axis=0), axis=1)
    a, sp = rise - rise.mean(), speed - speed.mean()
    lags = np.arange(-30, 31)
    xc = [np.dot(a[: len(a) - l], sp[l:]) if l >= 0 else np.dot(a[-l:], sp[: len(sp) + l]) for l in lags]
    st["lag_ms"] = float(lags[int(np.argmax(xc))] * 1000 / FS)
    # Within-subject linear decodability (log-envelope lags 0..150 ms); interleaved 10 s blocks, every 3rd held out.
    feats = np.hstack([np.roll(np.log(env + 1e-3), k, axis=0) for k in (0, 5, 10, 15)])[15:]
    y = z[15:, HAND]
    test = (np.arange(len(y)) // (10 * FS)) % 3 == 2
    mu, sd = feats[~test].mean(0), feats[~test].std(0) + 1e-9
    X = (feats - mu) / sd
    Xtr, Xte, ytr, yte = X[~test], X[test], y[~test], y[test]
    w = np.linalg.solve(Xtr.T @ Xtr + 10.0 * np.eye(X.shape[1]), Xtr.T @ (ytr - ytr.mean(0)))
    pred = Xte @ w + ytr.mean(0)
    keep = yte.std(0) > 1e-3
    r2 = 1 - ((yte - pred) ** 2).sum(0) / (((yte - yte.mean(0)) ** 2).sum(0) + 1e-9)
    st["ridge_r2"] = float(np.mean(r2[keep]))
    if rec["raw"] is not None:
        raw = rec["raw"].astype(np.float64)
        f, P = signal.welch(raw, fs=rec["fs_raw"], nperseg=256, axis=0)
        P = P.mean(1)
        P = P / P.sum()
        st["psd_freqs"], st["psd"] = f.tolist(), P.tolist()
        st["median_freq_hz"] = float(f[np.searchsorted(np.cumsum(P), 0.5)])
        st["clip_frac"] = float((np.abs(raw) >= 127).mean())
        r_raw = np.repeat(restim == 0, rec["fs_raw"] // FS)[: len(raw)]
        st["rest_small_frac"] = float((np.abs(raw[: len(r_raw)][r_raw]) <= 1).mean()) if r_raw.any() else None
    return st


SCALARS = ["rest_move_ratio", "participation_ratio", "autocorr_50ms", "autocorr_200ms", "lag_ms", "ridge_r2",
           "median_freq_hz", "clip_frac", "rest_small_frac"]


def summarise(stats):
    out = {}
    for k in SCALARS:
        v = [s[k] for s in stats if s.get(k) is not None]
        if v:
            out[k] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
    for k in ("amp_quantiles", "corr_vs_distance", "psd"):
        v = [s[k] for s in stats if k in s]
        if v:
            out[k] = np.mean(v, 0).tolist()
    if any("psd_freqs" in s for s in stats):
        out["psd_freqs"] = next(s["psd_freqs"] for s in stats if "psd_freqs" in s)
    return out


def gates(real, sim, margin=0.15):
    res = {}
    for k in SCALARS:
        if k in real and k in sim:
            lo, hi = real[k]["min"], real[k]["max"]
            pad = margin * max(hi - lo, abs(real[k]["mean"]) * 0.1, 1e-3)
            res[k] = {"real_range": [lo, hi], "real_mean": real[k]["mean"], "sim_mean": sim[k]["mean"],
                      "pass": bool(lo - pad <= sim[k]["mean"] <= hi + pad)}
    return res


def plot(real, sim, title, path):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    q = ["p5", "p25", "p50", "p75", "p95"]
    ax[0, 0].plot(q, real["amp_quantiles"], "o-", label="real")
    ax[0, 0].plot(q, sim["amp_quantiles"], "s-", label="sim")
    ax[0, 0].set_title("normalised envelope quantiles")
    ax[0, 0].legend()
    ax[0, 1].plot(real["corr_vs_distance"], "o-", label="real")
    ax[0, 1].plot(sim["corr_vs_distance"], "s-", label="sim")
    ax[0, 1].set_title("ring envelope correlation vs. electrode distance")
    ax[0, 1].set_xlabel("electrodes apart")
    keys = [k for k in SCALARS if k in real and k in sim and k not in ("median_freq_hz",)]
    for i, k in enumerate(keys):
        a = ax[0, 2] if k in ("lag_ms",) else ax[1, 0]
        if k == "lag_ms":
            a.bar([0, 1], [real[k]["mean"], sim[k]["mean"]], color=["C0", "C1"])
            a.errorbar([0], [real[k]["mean"]], yerr=[[real[k]["mean"] - real[k]["min"]], [real[k]["max"] - real[k]["mean"]]],
                       color="k", capsize=4)
            a.set_xticks([0, 1], ["real", "sim"])
            a.set_title("EMG-to-movement lag (ms, >0: EMG leads)")
            continue
        a.errorbar([i - 0.12], [real[k]["mean"]], yerr=[[real[k]["mean"] - real[k]["min"]], [real[k]["max"] - real[k]["mean"]]],
                   fmt="o", color="C0", capsize=4, label="real (range)" if i == 0 else None)
        a.plot([i + 0.12], [sim[k]["mean"]], "s", color="C1", label="sim" if i == 0 else None)
    ax[1, 0].set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
    ax[1, 0].set_title("scalar statistics")
    ax[1, 0].legend()
    if "psd" in real and "psd" in sim:
        ax[1, 1].semilogy(real["psd_freqs"], real["psd"], label="real")
        ax[1, 1].semilogy(sim["psd_freqs"], sim["psd"], label="sim")
        ax[1, 1].set_title("raw PSD (normalised)")
        ax[1, 1].set_xlabel("Hz")
        ax[1, 1].legend()
    else:
        ax[1, 1].axis("off")
    ax[1, 2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def run(real_name, sim_dir, out, max_subjects=None, bank=DATA_DIR / "latent_bank.h5", root=DATA_DIR):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    real = summarise([subject_stats(r) for r in load_real(real_name, max_subjects, bank, root)])
    sim = summarise([subject_stats(r) for r in load_sim(sim_dir, max_subjects)])
    g = gates(real, sim)
    (out / "realism.json").write_text(json.dumps({"real": real, "sim": sim, "gates": g}, indent=1))
    plot(real, sim, f"realism: {real_name} vs {sim_dir}", out / "realism.png")
    from rich.table import Table
    passed = sum(v["pass"] for v in g.values())
    t = Table(title=f"realism: {real_name} vs {sim_dir}  ({passed}/{len(g)} gates pass)", title_justify="left")
    for col in ("statistic", "real mean", "real range", "sim", "gate"):
        t.add_column(col, justify="left" if col == "statistic" else "right")
    for k, v in g.items():
        lo, hi = v["real_range"]
        t.add_row(k, f"{v['real_mean']:.3f}", f"{lo:.3f} .. {hi:.3f}", f"{v['sim_mean']:.3f}",
                  "[green]PASS" if v["pass"] else "[red]FAIL")
    console.print(t)
    log.info(f"corr vs distance  real {np.round(real['corr_vs_distance'], 2)}  sim {np.round(sim['corr_vs_distance'], 2)}")
    log.info(f"amp quantiles     real {np.round(real['amp_quantiles'], 3)}  sim {np.round(sim['amp_quantiles'], 3)}")
    log.info(f"wrote {out / 'realism.json'}, {out / 'realism.png'}")

