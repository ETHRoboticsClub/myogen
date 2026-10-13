"""Movement -> EMG benchmark on NinaPro DB5: does synthetic EMG help train joint-angle regressors?

    python scripts/benchmark.py simulate --name analytic_v0           # synthetic EMG of every DB5 recording
    python scripts/benchmark.py patterns --synthetic analytic_v0      # activation-pattern similarity (+ ceiling)
    python scripts/benchmark.py regress --synthetic analytic_v0       # regression protocols (real only without it)

simulate  Each subject's glove motion (all exercises, concatenated) through myogen with the Myo profile. The EMG is
          saved per recording as data/synthetic/<name>/sN/SN_EK.npy, aligned sample by sample with the real file.
patterns  Per subject, the (gesture x channel) activation pattern: mean log RMS during each gesture minus rest,
          centred per channel across gestures (channel gains drop out). Similarity of two patterns: correlation,
          maximised over armband rotations (both Myo rings shifted together). Real vs real between subjects is the
          ceiling of a simulator that knows nothing about the subject; simulated vs real (same subject) the score.
regress   Joint-angle regression from EMG features (log RMS and log waveform length per channel, 200 ms windows
          every 50 ms, 250 ms of context). NinaPro split: repetitions 1, 3, 4, 6 train, 2 and 5 test. Features are
          standardised per subject and domain with that subject's training repetitions (unlabelled EMG only).
          Protocols, all tested on the real test repetitions of each subject:
            real            real training repetitions of the subject (reference)
            real_k          only k real repetitions (k = 1, 2: the low-data regime)
            real_cross      real data of the other subjects
            shuffled        real, with the training labels shifted in time (chance level)
            sim             the subject's simulated training repetitions only
            sim_cross       simulated data of all subjects (the test repetitions of the subject excluded)
            real_k+sim      augmentation: k real repetitions plus the subject's simulated training repetitions
            pretrain_k      mlp only: trained on sim_cross, then fine-tuned on k real repetitions
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import dataclasses
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from ninapro import glove_to_angles, subject_files
except ImportError:                                   # run as python -m scripts.benchmark
    from scripts.ninapro import glove_to_angles, subject_files

DB5 = Path("data/real/ninapro_db5")
SYNTHETIC = Path("data/synthetic")
REPORTS = Path("reports/benchmark")
FS = 200.0
WIN, STEP, CONTEXT = 40, 10, 5                        # samples (200 ms, 50 ms), windows of context
ALL_SUBJECTS = list(range(1, 11))
TRAIN_REPS, TEST_REPS, FEW_REPS = (1, 3, 4, 6), (2, 5), {1: (1,), 2: (1, 4)}
TARGETS = ["pro_sup_r", "deviation_r", "flexion_r", "cmc_abduction_r", "mp_flexion_r",
           "mcp2_flexion_r", "mcp3_flexion_r", "mcp4_flexion_r", "mcp5_flexion_r", "mcp2_abduction_r"]


# -- simulate ----------------------------------------------------------------------------------------------------

def simulate_subject(job):
    try:
        return _simulate_subject(*job)
    except Exception as e:                            # one subject failing must not stop the others
        return f"{job[0].name}: FAILED {type(e).__name__}: {e}"


def _simulate_subject(folder, out, engine, electrodes, seed):
    import logging

    import myogen as mg
    logging.basicConfig(level=logging.WARNING, format=f"{folder.name}: %(message)s")
    t0 = time.time()
    recs = subject_files(folder)
    angles = [glove_to_angles(r, recs) for r in recs]
    motion = mg.JointAdapter()({k: np.concatenate([a[k] for a in angles]) for k in angles[0]}, FS,
                               rest=np.concatenate([r["restimulus"] == 0 for r in recs]))
    if electrodes.endswith((".yml", ".yaml")):
        electrodes = mg.Electrodes.load(electrodes)
    sim = mg.Simulator(electrodes, engine=engine, subject=dataclasses.replace(mg.Subject(), seed=seed))
    emg = sim.run(motion)
    out.mkdir(parents=True, exist_ok=True)
    start = 0
    for r in recs:
        n = len(r["emg"])
        np.save(out / f"{r['path'].stem.split('_A')[0]}.npy", emg.emg[start:start + n].astype(np.float32))
        start += n
    (out / "qc.json").write_text(json.dumps(emg.qc))
    return (f"{folder.name}: {start / FS:.0f} s simulated in {(time.time() - t0) / 60:.1f} min, tracking RMSE "
            f"{emg.qc['rmse_deg']:.2f} deg, {len(emg.qc['recovered_ticks'])} recoveries")


def cmd_simulate(a):
    out = SYNTHETIC / a.name
    jobs = [(DB5 / f"s{s}", out / f"s{s}", a.engine, a.electrodes, 1000 + s) for s in a.subjects]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"engine": a.engine, "electrodes": a.electrodes, "commit": commit,
                                                 "subject": "Subject() with seed 1000 + subject"}))
    with Pool(min(a.workers, len(jobs))) as pool:
        for msg in pool.imap_unordered(simulate_subject, jobs):
            print(msg, flush=True)


# -- data ----------------------------------------------------------------------------------------------------------

def load_subject(s, synthetic=None):
    """{"real": Windows, "sim": Windows or None} of DB5 subject s."""
    recs = subject_files(DB5 / f"s{s}")
    synthetic = synthetic if synthetic and (SYNTHETIC / synthetic / f"s{s}" / "qc.json").exists() else None
    out = {"real": [], "sim": [] if synthetic else None}
    for r in recs:
        ang = glove_to_angles(r, recs)
        y = np.stack([ang[k] for k in TARGETS], 1)
        meta = (y, r["restimulus"], _fill_reps(r["rerepetition"]), r["exercise"])
        out["real"].append(windows(r["emg"], *meta))
        if synthetic:
            out["sim"].append(windows(np.load(SYNTHETIC / synthetic / f"s{s}" / f"S{s}_E{r['exercise']}.npy"), *meta))
    return {k: (_concat(v) if v is not None else None) for k, v in out.items()}


def windows(emg, y, restimulus, rep, exercise):
    """Features (n, CONTEXT * 2C) and, at each window's last sample: labels, gesture id, repetition."""
    x = np.asarray(emg, float)
    ends = np.arange(WIN, len(x) + 1, STEP)
    c2 = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x ** 2]), 0)
    cw = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), np.abs(np.diff(x, axis=0, prepend=x[:1]))]), 0)
    f = np.hstack([np.log(np.sqrt((c2[ends] - c2[ends - WIN]) / WIN) + 1e-3),
                   np.log((cw[ends] - cw[ends - WIN]) / WIN + 1e-3)])
    ctx = np.hstack([f[np.maximum(np.arange(len(f)) - k, 0)] for k in range(CONTEXT)])
    i = ends - 1
    gesture = np.where(restimulus[i] > 0, 100 * exercise + restimulus[i], 0)
    return {"x": ctx, "logrms": f[:, :x.shape[1]], "y": y[i], "gesture": gesture, "rep": rep[i]}


def _fill_reps(rerep):
    """Repetition of every sample: rest belongs to the preceding movement (leading rest: the first one)."""
    r = np.asarray(rerep).ravel().astype(int)
    idx = np.where(r > 0, np.arange(len(r)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = r[idx]
    out[out == 0] = r[r > 0][0] if (r > 0).any() else 1
    return out


def _concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _standardise(d):
    """Standardise d["x"] with the statistics of its training repetitions (in place)."""
    tr = np.isin(d["rep"], TRAIN_REPS)
    mu, sd = d["x"][tr].mean(0), d["x"][tr].std(0) + 1e-6
    d["x"] = (d["x"] - mu) / sd
    return d


def _select(d, reps):
    m = np.isin(d["rep"], reps)
    return {k: v[m] for k, v in d.items()}


def _join(parts):
    return _concat([p for p in parts if p is not None and len(p["x"])])


# -- patterns ----------------------------------------------------------------------------------------------------

def pattern(d):
    """(gestures, channels) mean log RMS per gesture minus rest, centred per channel; and the gesture ids."""
    g = np.unique(d["gesture"][d["gesture"] > 0])
    rest = d["logrms"][d["gesture"] == 0].mean(0)
    P = np.stack([d["logrms"][d["gesture"] == k].mean(0) - rest for k in g])
    return P - P.mean(0), g


def similarity(A, B, rotate=True):
    """(correlation, rotation in channels): max over shifts of both 8-channel rings of B."""
    best = (-2.0, 0)
    for k in range(8 if rotate else 1):
        Bk = np.hstack([np.roll(B[:, :8], k, 1), np.roll(B[:, 8:], k, 1)])
        best = max(best, (float(np.corrcoef(A.ravel(), Bk.ravel())[0, 1]), k))
    return best


def cmd_patterns(a):
    data = {s: load_subject(s, a.synthetic if s in a.subjects else None) for s in ALL_SUBJECTS}
    P = {s: {dom: pattern(d) for dom, d in data[s].items() if d is not None} for s in ALL_SUBJECTS}
    rows, report = [], {"real_vs_real": {}, "sim_vs_real": {}}
    for s in a.subjects:
        Ps, gs = P[s]["real"]
        others = []
        for t in ALL_SUBJECTS:
            if t != s:
                common = np.intersect1d(gs, P[t]["real"][1])
                others.append([similarity(Ps[np.isin(gs, common)], P[t]["real"][0][np.isin(P[t]["real"][1], common)],
                                          rot)[0] for rot in (False, True)])
        report["real_vs_real"][s] = np.mean(others, 0).tolist()
        row = [s, *report["real_vs_real"][s]]
        if "sim" in P[s]:
            Pm, gm = P[s]["sim"]
            fixed, (rotated, shift) = similarity(Ps, Pm, False), similarity(Ps, Pm)
            per_ex = {e: similarity(Ps[gs // 100 == e], np.hstack([np.roll(Pm[gm // 100 == e, :8], shift, 1),
                                                                  np.roll(Pm[gm // 100 == e, 8:], shift, 1)]),
                                    False)[0] for e in (1, 2, 3)}
            report["sim_vs_real"][s] = {"fixed": fixed[0], "rotated": rotated, "shift": shift, "per_exercise": per_ex}
            row += [fixed[0], rotated, shift, *per_ex.values()]
        elif a.synthetic:
            row += ["-"] * 6
        rows.append(row)
    head = ["subject", "real-real", "real-real rot"] + (["sim-real", "sim-real rot", "shift", "E1", "E2", "E3"]
                                                         if a.synthetic else [])
    _table(head, rows)
    _save(f"patterns_{a.synthetic or 'real'}{_tag(a)}.json", report)


# -- regress -----------------------------------------------------------------------------------------------------

def fit_ridge(train, alpha=10.0):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha).fit(train["x"], train["y"]).predict


class MLP:
    def __init__(self, n_in, n_out, seed=0):
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                       torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                       torch.nn.Linear(256, n_out))
        self.y_mu = self.y_sd = None

    def fit(self, d, steps=3000, lr=1e-3, batch=512, seed=0):
        torch = self.torch
        if self.y_mu is None:
            self.y_mu, self.y_sd = d["y"].mean(0), d["y"].std(0) + 1e-6
        x = torch.tensor(d["x"], dtype=torch.float32)
        y = torch.tensor((d["y"] - self.y_mu) / self.y_sd, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        g = torch.Generator().manual_seed(seed)
        self.net.train()
        for _ in range(steps):
            i = torch.randint(len(x), (batch,), generator=g)
            loss = torch.nn.functional.mse_loss(self.net(x[i]), y[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def predict(self, x):
        self.net.eval()
        with self.torch.no_grad():
            return self.net(self.torch.tensor(x, dtype=self.torch.float32)).numpy() * self.y_sd + self.y_mu


def fit_mlp(train, pretrained=None):
    if pretrained is not None:
        model = MLP(train["x"].shape[1], train["y"].shape[1])
        model.net.load_state_dict(pretrained.net.state_dict())
        model.y_mu, model.y_sd = pretrained.y_mu, pretrained.y_sd
        return model.fit(train, steps=1000, lr=3e-4).predict
    return MLP(train["x"].shape[1], train["y"].shape[1]).fit(train).predict


def score(pred, y):
    """(mean Pearson r, mean R2) over the target joints that move in the test set."""
    r, r2 = [], []
    for j in range(y.shape[1]):
        if y[:, j].std() < 0.02:
            continue
        r.append(np.corrcoef(pred[:, j], y[:, j])[0, 1])
        r2.append(1 - np.sum((pred[:, j] - y[:, j]) ** 2) / np.sum((y[:, j] - y[:, j].mean()) ** 2))
    return float(np.mean(r)), float(np.mean(r2))


def cmd_regress(a):
    import torch
    torch.set_num_threads(a.threads)
    data = {s: {k: _standardise(v) if v is not None else None
                for k, v in load_subject(s, a.synthetic if s in a.subjects else None).items()}
            for s in ALL_SUBJECTS}
    results = {}
    for s in a.subjects:
        real, sim = data[s]["real"], data[s]["sim"]
        test = _select(real, TEST_REPS)
        train = {"real": _select(real, TRAIN_REPS),
                 "real_cross": _join([data[t]["real"] for t in ALL_SUBJECTS if t != s])}
        shuffled = dict(train["real"])
        shuffled["y"] = np.roll(shuffled["y"], len(shuffled["y"]) // 2, 0)
        train["shuffled"] = shuffled
        for k, reps in FEW_REPS.items():
            train[f"real_{k}"] = _select(real, reps)
        if sim is not None:
            train["sim"] = _select(sim, TRAIN_REPS)
            train["sim_cross"] = _join([train["sim"]] + [data[t]["sim"] for t in a.subjects if t != s])
            for k in FEW_REPS:
                train[f"real_{k}+sim"] = _join([train[f"real_{k}"], train["sim"]])
        results[s] = {}
        for model in a.models:
            fit = fit_ridge if model == "ridge" else fit_mlp
            for name, d in train.items():
                results[s][f"{model}:{name}"] = score(fit(d)(test["x"]), test["y"])
            if model == "mlp" and sim is not None:
                base = MLP(train["sim_cross"]["x"].shape[1], len(TARGETS)).fit(train["sim_cross"])
                for k in FEW_REPS:
                    results[s][f"mlp:pretrain_{k}"] = score(fit_mlp(train[f"real_{k}"], base)(test["x"]), test["y"])
        print(f"s{s}: " + "  ".join(f"{k} {v[0]:.2f}" for k, v in results[s].items()), flush=True)
    keys = list(dict.fromkeys(k for s in a.subjects for k in results[s]))
    rows = []
    for k in keys:
        v = [results[s][k] for s in a.subjects if k in results[s]]
        rows.append([k, *np.mean(v, 0), float(np.std([x[0] for x in v])), str(len(v))])
    _table(["protocol", "r", "R2", "r sd", "subjects"], rows)
    _save(f"regress_{a.synthetic or 'real'}{_tag(a)}.json", {"subjects": a.subjects, "targets": TARGETS, "results": results})


# -- output ------------------------------------------------------------------------------------------------------

def _table(head, rows):
    print("".join(f"{h:>16s}" for h in head))
    for r in rows:
        print("".join(f"{v:>16.3f}" if isinstance(v, float) else f"{str(v):>16s}" for v in r))


def _tag(a):
    return "" if a.subjects == ALL_SUBJECTS else f"_s{'-'.join(map(str, a.subjects))}"


def _save(name, obj):
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote {REPORTS / name}")


def _subjects(text):
    out = []
    for part in text.split(","):
        lo, hi = part.split("-") if "-" in part else (part, part)
        out += range(int(lo), int(hi) + 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("simulate")
    p.add_argument("--name", required=True)
    p.add_argument("--engine", default="analytic", choices=("analytic", "biomime"))
    p.add_argument("--electrodes", default="myo", help="preset name or profile YAML")
    p.add_argument("--workers", type=int, default=10)
    for name in ("patterns", "regress"):
        q = sub.add_parser(name)
        q.add_argument("--synthetic", help="name of a simulated set in data/synthetic")
    sub.choices["regress"].add_argument("--models", default="ridge,mlp", type=lambda s: s.split(","))
    sub.choices["regress"].add_argument("--threads", type=int, default=8)
    for q in sub.choices.values():
        q.add_argument("--subjects", default="1-10", type=_subjects, help="e.g. 1-10 or 1,2,8")
    a = ap.parse_args()
    {"simulate": cmd_simulate, "patterns": cmd_patterns, "regress": cmd_regress}[a.cmd](a)


if __name__ == "__main__":
    main()
