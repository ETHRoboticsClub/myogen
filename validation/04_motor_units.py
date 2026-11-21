import numpy as np
from common import REFS, Forearm, Report
from scipy import signal, stats

import myogen as mg
from myogen.params import EMGParams
from myogen.simulation.emg import EMGModel

r = REFS["motor_units"]
f = Forearm()
over = mg.Electrodes([mg.Electrode(60, 7, ied_mm=10)])                   # bipolar, over EDC (test_simulator.py)
emg = EMGModel(f.model, f.subject, over, EMGParams())
units, mu = emg.pool, f.muscle(r["muscle"])
own = np.flatnonzero(units.muscle == mu)
ticks = int(r["seconds"] * 100)

rates, cvs = [], []
for level in r["drives"]:
    drive = np.zeros((ticks, f.model.nu))
    drive[:, mu] = level
    spikes = units.spikes(drive, 100.0, np.random.default_rng(0))
    times, fired = spikes.times, spikes.units
    trains = [times[fired == u] for u in own if (fired == u).sum() > 3]
    rates.append([1 / np.median(np.diff(t)) for t in trains])           # while firing
    cvs += [np.std(np.diff(t)) / np.mean(np.diff(t)) for t in trains]

# The EMG statistics depend on where the few units nearest the electrode sit: median over subjects (seeds).
rms, kurt, median = (np.zeros((r["seeds"], len(r["drives"]))) for _ in range(3))
for s in range(r["seeds"]):
    emg_s = EMGModel(f.model, mg.Subject(seed=s), over, EMGParams())
    for k, level in enumerate(r["drives"]):
        x = emg_s.generate(f.activity({r["muscle"]: level}, ticks), np.random.default_rng(s))[:, 0]
        rms[s, k], kurt[s, k] = np.sqrt(np.mean(x ** 2)), stats.kurtosis(x)
        fr, p = signal.welch(x, emg.fs, nperseg=1024)
        median[s, k] = fr[np.searchsorted(np.cumsum(p), p.sum() / 2)]
rms, kurt, median = (list(np.median(v, axis=0)) for v in (rms, kurt, median))

report = Report(f"Motor units of {r['muscle']} (bipolar 10 mm over it)")
report.check("rate at recruitment, lowest (Hz)", min(min(x) for x in rates if x), *r["recruitment_rate_hz"])
report.check("peak rate at full drive (Hz)", max(rates[-1]), *r["peak_rate_hz"])
report.check("ISI coefficient of variation (median)", np.median(cvs), *r["isi_cv"])
report.check("drive recruiting the last unit", units.threshold[own].max(), *r["last_recruited_drive"])
fit = slice(1, -1)
b = np.polyfit(np.log(r["drives"][fit]), np.log(rms[fit]), 1)[0]
report.check("RMS ~ drive^b: b", b, *r["rms_exponent"])
report.check("median frequency at 40 % drive (Hz)", median[r["drives"].index(0.4)], *r["median_hz"])
report.check("excess kurtosis falls: low - high drive", kurt[0] - kurt[-1], 0, np.inf)
report.check("excess kurtosis at full drive", kurt[-1], -1, r["max_excess_kurtosis_high"])
print("  drive " + " ".join(f"{d:6.2f}" for d in r["drives"]))
print("  units " + " ".join(f"{len(x):6d}" for x in rates) + f"   of {len(own)}")
print("  rms   " + " ".join(f"{v:6.3f}" for v in rms))
print("  kurt  " + " ".join(f"{v:6.2f}" for v in kurt))
report.show()
