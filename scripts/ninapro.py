"""NinaPro DB1 / DB5 recordings: loading and glove -> model joint angles (shared by the scripts).

Glove -> joint angles: the CyberGlove sensors are normalised per subject (1st-99th percentile over all the
subject's files, rest-centred for the wrist), turned into a hand posture (digit closures, thumb abduction, spread,
wrist) and mapped to the model's joints with fixed finger coupling. Pronation / supination comes from the
accelerometer inside the Myo armbands (DB5: gravity's roll around the forearm, relative to rest) or the movement
labels (DB1, exercise 2); during labelled rest it returns to the rest reference (the subject is relaxed; this also
removes accelerometer transients). DB1's wrist-pitch sensor duplicates sensor 12, so its wrist flexion also comes
from the labels. This is a heuristic retargeting: the glove is uncalibrated.
"""
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat

# CyberGlove 22 sensor indices (checked against the single-finger movements of exercise 1).
THUMB_ROT, THUMB_MCP, THUMB_IP, THUMB_ABD = 0, 1, 2, 3
FINGERS = [(4, 5, 6), (7, 8, 9), (11, 12, 13), (15, 16, 17)]           # index, middle, ring, little: MCP PIP DIP
FINGER_ABD = (10, 14, 18)
WRIST_PITCH, WRIST_YAW = 20, 21
# Exercise-2 movement id -> (posture channel, target value), for channels the glove does not measure.
EX2_TARGETS = {9: ("pro_sup", 0.7), 10: ("pro_sup", -0.7), 11: ("pro_sup", 0.7), 12: ("pro_sup", -0.7),
               13: ("wrist_flex", 0.7), 14: ("wrist_flex", -0.7), 17: ("wrist_flex", -0.6)}


def load_mat(path):
    d = loadmat(path)
    db5 = "acc" in d
    rec = {"path": Path(path), "dataset": "db5" if db5 else "db1", "fs": 200.0 if db5 else 100.0,
           "emg": d["emg"].astype(np.float32), "glove": d["glove"].astype(float),
           "restimulus": d["restimulus"].ravel().astype(int),
           "rerepetition": d["rerepetition"].ravel().astype(int),
           "exercise": int(Path(path).stem.split("_E")[1][0])}            # the file name is authoritative
    if db5:
        rec["acc"] = d["acc"].astype(float)
    return rec


def glove_to_angles(rec, subject_recs):
    """{model joint: (T,) rad} for one recording; subject_recs (all files of the subject) set the normalisation."""
    glove_all = np.concatenate([r["glove"] for r in subject_recs])
    lo, hi = np.percentile(glove_all, [1, 99], axis=0)
    span = np.maximum(hi - lo, 1e-6)
    rest_med = np.median(np.concatenate([r["glove"][r["restimulus"] == 0] for r in subject_recs]), axis=0)
    fs = rec["fs"]
    g = _lowpass(rec["glove"], 3.0, fs)
    n = np.clip((g - lo) / span, 0, 1)                                 # 0..1 per sensor
    c = np.clip((g - rest_med) / (span / 2), -1, 1)                    # rest-centred, -1..1

    thumb = np.clip(1 - n[:, [THUMB_MCP, THUMB_IP]].mean(1), 0, 1)    # thumb flexion lowers sensors 1, 2
    fingers = [np.clip(n[:, list(idx)].mean(1), 0, 1) for idx in FINGERS]
    thumb_abd = np.clip(0.5 * (n[:, THUMB_ROT] + 1 - n[:, THUMB_ABD]), 0, 1)
    spread = np.clip(1 - n[:, list(FINGER_ABD)].mean(1), 0, 1)
    pitch_broken = rec["dataset"] == "db1" and np.corrcoef(glove_all[:, WRIST_PITCH], glove_all[:, 12])[0, 1] > 0.95
    wrist_flex = _from_labels(rec, "wrist_flex") if pitch_broken else c[:, WRIST_PITCH]
    wrist_dev = c[:, WRIST_YAW]
    pro_sup = _pro_sup(rec, subject_recs)

    # Posture -> joints with fixed coupling (MCP takes 87.5 % of a finger's closure, DIP = 0.725 PIP).
    lerp = lambda a, b, t: a + (b - a) * t
    q = {"pro_sup_r": 1.3 * pro_sup, "deviation_r": np.where(wrist_dev > 0, 0.4, 0.17) * wrist_dev,
         "flexion_r": 0.75 * wrist_flex * np.where(wrist_flex > 0, 1 - 0.55 * np.mean(fingers, 0), 1.0),
         "cmc_flexion_r": lerp(0.3, -0.6, thumb), "cmc_abduction_r": lerp(-0.4, 0.7, thumb_abd),
         "mp_flexion_r": lerp(0.2, -0.6, thumb), "ip_flexion_r": lerp(0.2, -1.0, thumb)}
    for k, (f, closure) in enumerate(zip("2345", fingers)):
        q[f"mcp{f}_flexion_r"] = 1.45 * np.clip(closure * 0.875 / 0.9, 0, 1)
        q[f"mcp{f}_abduction_r"] = 0.22 * [-1.0, -0.2, 0.6, 1.0][k] * (spread - 0.3) * (1 - closure)
        q[f"pm{f}_flexion_r"] = 1.5 * closure
        q[f"md{f}_flexion_r"] = 0.725 * 1.5 * closure
    return q


def _pro_sup(rec, subject_recs):
    """Pronation / supination in [-1, 1]: Myo accelerometer roll (DB5) or the movement labels (DB1)."""
    if "acc" not in rec:
        return _from_labels(rec, "pro_sup")
    acc_rest = np.concatenate([r["acc"][r["restimulus"] == 0] for r in subject_recs])
    roll_ref = np.median(np.arctan2(acc_rest[:, 1], acc_rest[:, 2]))
    gain = 1.0
    ex2 = [r for r in subject_recs if r["exercise"] == 2]
    if ex2:
        # The armband follows only part of the radius's rotation: scale the labelled pro/supinations to +-0.7.
        roll2 = np.arctan2(ex2[0]["acc"][:, 1], ex2[0]["acc"][:, 2]) - roll_ref
        peaks = [np.percentile(np.abs(roll2[ex2[0]["restimulus"] == mv]), 80) for mv in (9, 10, 11, 12)
                 if (ex2[0]["restimulus"] == mv).any()]
        if peaks and np.mean(peaks) > 1e-3:
            gain = np.clip(0.7 * 1.3 / np.mean(peaks), 1.0, 8.0)
    acc = _lowpass(rec["acc"], 3.0, rec["fs"])
    pro_sup = np.clip((np.unwrap(np.arctan2(acc[:, 1], acc[:, 2])) - roll_ref) * gain / 1.3, -1, 1)
    at_rest = np.clip(_lowpass((rec["restimulus"] == 0).astype(float), 1.5, rec["fs"]), 0, 1)
    return pro_sup * (1 - at_rest)                                     # relaxed forearm at rest


def _from_labels(rec, channel):
    """A smooth trajectory for a channel the glove does not measure, from the exercise-2 movement labels."""
    target = np.zeros(len(rec["restimulus"]))
    if rec["exercise"] == 2:
        for mv, (ch, value) in EX2_TARGETS.items():
            if ch == channel:
                target[rec["restimulus"] == mv] = value
    return _lowpass(target, 1.5, rec["fs"])


def _lowpass(x, fc, fs):
    return signal.sosfiltfilt(signal.butter(2, fc, fs=fs, output="sos"), x, axis=0)


def subject_files(folder):
    """All recordings of one subject folder (e.g. data/real/ninapro_db5/s1), in exercise order."""
    return [load_mat(p) for p in sorted(Path(folder).glob("*.mat"))]
