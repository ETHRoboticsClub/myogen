"""NinaPro DB1 / DB5 loading and CyberGlove -> posture-latent retargeting.

    myogen ninapro-bank      # -> paths.latent_bank (real intent trajectories for the generator)

Datasets (copied into data/real/):
  DB1: 27 subjects, 10 Otto Bock electrodes (8 ring at the radio-humeral joint + FDS + EDC spots), EMG is the
       electrodes' rectified/smoothed output at 100 Hz. CyberGlove 22 at 100 Hz. Sensor 20 duplicates 12.
  DB5: 10 subjects, two Myo armbands (16 raw channels, 200 Hz, int8), the second ring nearer the hand and
       rotated 22.5 deg. CyberGlove 22 (upsampled to 200 Hz), Myo accelerometer, forearm circumference.
  Both: exercise 1 = 12 finger movements, 2 = 17 postures/wrist movements, 3 = 23 grasps, 6 or 10 repetitions.

Glove sensors (standard CyberGlove 22 order, checked against single-finger movements of exercise 1):
  0 thumb rotation, 1 thumb MCP, 2 thumb IP, 3 thumb-index abduction,
  4-6 index MCP/PIP/DIP, 7-9 middle MCP/PIP/DIP, 10 middle-index abd, 11-13 ring, 14 ring-middle abd,
  15-17 little, 18 little-ring abd, 19 palm arch, 20 wrist pitch, 21 wrist yaw.

Latent channels the glove does not measure are recovered or imputed and flagged in `imputed`:
  * pro_sup:   DB5 from the Myo accelerometer (roll of gravity around the forearm axis, gain calibrated per subject
               on the labelled pronation/supination movements); DB1 from movement labels.
  * wrist_flex: DB1 from movement labels (its wrist-pitch sensor is a copy of sensor 12).
"""
import logging
from pathlib import Path

import h5py
import numpy as np
from scipy import signal
from scipy.io import loadmat

from .config import user_data_dir
from .motion import LATENT_NAMES, N_LATENT

log = logging.getLogger(__name__)
DATA_DIR = user_data_dir() / "real"     # default; commands pass paths.ninapro / paths.latent_bank
FS = 100  # latent bank rate = generator control rate

THUMB_ROT, THUMB_MCP, THUMB_IP, THUMB_ABD = 0, 1, 2, 3
FINGERS = {"index": (4, 5, 6), "middle": (7, 8, 9), "ring": (11, 12, 13), "little": (15, 16, 17)}
FINGER_ABD = (10, 14, 18)
WRIST_PITCH, WRIST_YAW = 20, 21

# Exercise-2 movement ids -> (latent channel, target value) used to impute unmeasured channels.
EX2_TARGETS = {
    9: ("pro_sup", 0.7), 10: ("pro_sup", -0.7), 11: ("pro_sup", 0.7), 12: ("pro_sup", -0.7),
    13: ("wrist_flex", 0.7), 14: ("wrist_flex", -0.7), 17: ("wrist_flex", -0.6),
}


def load_mat(path):
    d = loadmat(path)
    db5 = "acc" in d
    rec = {
        "dataset": "db5" if db5 else "db1",
        "emg": d["emg"].astype(np.float32),
        "glove": d["glove"].astype(np.float64),
        "restimulus": d["restimulus"].ravel().astype(np.int16),
        "rerepetition": d["rerepetition"].ravel().astype(np.int16),
        "fs": 200 if db5 else 100,
        "subject": int(d["subject"].item()),
        "exercise": int(Path(path).stem.split("_E")[1][0]),   # file name is authoritative (DB5 attrs differ)
    }
    if db5:
        rec["acc"] = d["acc"].astype(np.float64)
        rec["circumference_cm"] = float(d["circumference"].item())
    return rec


def subject_files(dataset, subject, root=DATA_DIR):
    d = root / f"ninapro_{dataset}" / f"s{subject}"
    return sorted(d.glob("*.mat"))


def list_subjects(dataset, root=DATA_DIR):
    return sorted(int(p.name[1:]) for p in (root / f"ninapro_{dataset}").glob("s*") if p.is_dir())


def _lowpass(x, fc, fs, order=2):
    sos = signal.butter(order, fc, fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def _to_fs(x, fs):
    if fs == FS:
        return x
    return signal.resample_poly(x, FS, fs, axis=0)


def _label_track(restim, fs, mapping, channel):
    """Smooth target trajectory for `channel` from movement labels (min-jerk-like via low-pass)."""
    target = np.zeros(len(restim))
    for mv, (ch, val) in mapping.items():
        if ch == channel:
            target[restim == mv] = val
    return _lowpass(target, 1.5, fs)


def retarget_subject(recs):
    """Glove (+acc) of all exercises of one subject -> list of per-exercise dicts at FS."""
    glove_all = np.concatenate([r["glove"] for r in recs])
    lo, hi = np.percentile(glove_all, [1, 99], axis=0)
    span = np.maximum(hi - lo, 1e-6)
    rest_all = np.concatenate([r["glove"][r["restimulus"] == 0] for r in recs])
    rest_med = np.median(rest_all, axis=0)
    db1_pitch_broken = np.corrcoef(glove_all[:, WRIST_PITCH], glove_all[:, 12])[0, 1] > 0.95
    roll_ref, roll_gain = None, 1.0
    if "acc" in recs[0]:
        acc_rest = np.concatenate([r["acc"][r["restimulus"] == 0] for r in recs])
        roll_ref = np.median(np.arctan2(acc_rest[:, 1], acc_rest[:, 2]))
        # A proximal armband only follows part of the radius's rotation: scale so the labelled
        # pronation/supination movements of exercise 2 reach the same +-0.7 latent as the imputed DB1 tracks.
        ex2 = [r for r in recs if r["exercise"] == 2]
        if ex2:
            r2 = ex2[0]
            roll2 = np.arctan2(r2["acc"][:, 1], r2["acc"][:, 2]) - roll_ref
            peaks = [np.percentile(np.abs(roll2[r2["restimulus"] == mv]), 80) for mv in (9, 10, 11, 12)
                     if (r2["restimulus"] == mv).any()]
            if peaks and np.mean(peaks) > 1e-3:
                roll_gain = np.clip(0.7 * 1.3 / np.mean(peaks), 1.0, 8.0)

    out = []
    for r in recs:
        fs = r["fs"]
        g = _lowpass(r["glove"], 3.0, fs)
        n = np.clip((g - lo) / span, 0, 1)                          # 0..1 per sensor
        c = np.clip((g - rest_med) / (span / 2), -1, 1)             # rest-centred, -1..1
        z = np.zeros((len(g), N_LATENT))
        imputed = np.zeros(N_LATENT, bool)
        z[:, 0] = 1 - n[:, [THUMB_MCP, THUMB_IP]].mean(1)           # thumb flexion lowers 1, 2
        for k, idx in enumerate(FINGERS.values()):
            z[:, 1 + k] = n[:, idx].mean(1)
        z[:, 5] = 0.5 * (n[:, THUMB_ROT] + 1 - n[:, THUMB_ABD])
        z[:, 6] = 1 - n[:, FINGER_ABD].mean(1)
        if r["dataset"] == "db1" and db1_pitch_broken:
            z[:, 7] = _label_track(r["restimulus"], fs, EX2_TARGETS, "wrist_flex") if r["exercise"] == 2 else 0
            imputed[7] = True
        else:
            z[:, 7] = c[:, WRIST_PITCH]
        z[:, 8] = c[:, WRIST_YAW]
        if roll_ref is not None:
            acc = _lowpass(r["acc"], 3.0, fs)
            roll = np.unwrap(np.arctan2(acc[:, 1], acc[:, 2])) - roll_ref
            z[:, 9] = np.clip(roll * roll_gain / 1.3, -1, 1)        # pro_sup_r spans +-1.3 rad of latent
        else:
            z[:, 9] = _label_track(r["restimulus"], fs, EX2_TARGETS, "pro_sup") if r["exercise"] == 2 else 0
            imputed[9] = True
        z[:, :7] = np.clip(z[:, :7], 0, 1)
        z[:, 7:] = np.clip(z[:, 7:], -1, 1)
        out.append({
            "latent": _to_fs(z, fs).astype(np.float32),
            "restimulus": r["restimulus"][:: fs // FS],
            "rerepetition": r["rerepetition"][:: fs // FS],
            "imputed": imputed,
            "dataset": r["dataset"], "subject": r["subject"], "exercise": r["exercise"],
        })
    return out


def build_bank(out=DATA_DIR / "latent_bank.h5", root=DATA_DIR):
    """Latent bank from the raw NinaPro recordings under `root` (ninapro_db1/, ninapro_db5/)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with h5py.File(out, "w") as f:
        f.attrs["latent_names"] = LATENT_NAMES
        f.attrs["fs"] = FS
        for ds in ("db1", "db5"):
            for s in list_subjects(ds, root):
                recs = [load_mat(p) for p in subject_files(ds, s, root)]
                for e in retarget_subject(recs):
                    g = f.create_group(f"{ds}/s{s:02d}/e{e['exercise']}")
                    for k in ("latent", "restimulus", "rerepetition", "imputed"):
                        g.create_dataset(k, data=e[k], compression="gzip")
                    n += 1
                    log.info(f"{ds} s{s:02d} e{e['exercise']}: {len(e['latent']) / FS:6.1f} s  "
                             f"imputed={[LATENT_NAMES[i] for i in np.flatnonzero(e['imputed'])]}")
    log.info(f"wrote {out} ({n} exercise recordings)")


class LatentBank:
    """Random access to real intent trajectories for the generator."""

    def __init__(self, path=DATA_DIR / "latent_bank.h5", datasets=("db1", "db5")):
        if not Path(path).exists():
            raise FileNotFoundError(f"latent bank {path} missing: `myogen download latent-bank`, or build it from "
                                    f"NinaPro data with `myogen ninapro-bank`")
        self.items = []
        with h5py.File(path, "r") as f:
            for ds in datasets:
                if ds not in f:
                    continue
                for s in f[ds]:
                    for e in f[ds][s]:
                        g = f[ds][s][e]
                        self.items.append({"key": f"{ds}/{s}/{e}", "latent": g["latent"][:],
                                           "restimulus": g["restimulus"][:], "imputed": g["imputed"][:]})
        if not self.items:
            raise FileNotFoundError(f"empty latent bank {path}: rebuild it with `myogen ninapro-bank`")
        self.weights = np.array([len(it["latent"]) for it in self.items], float)
        self.weights /= self.weights.sum()

    def sample(self, rng, n_samples):
        """A random contiguous window (n_samples at FS) -> (latent, restimulus, source key, start)."""
        while True:
            it = self.items[rng.choice(len(self.items), p=self.weights)]
            if len(it["latent"]) > n_samples:
                break
        i0 = int(rng.integers(0, len(it["latent"]) - n_samples))
        return it["latent"][i0:i0 + n_samples].copy(), it["restimulus"][i0:i0 + n_samples].copy(), it["key"], i0
