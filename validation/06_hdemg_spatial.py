import urllib.request
from pathlib import Path

import numpy as np
from common import REFS, Forearm, Report
from scipy import signal

import myogen as mg
from myogen.params import EMGParams, MotorUnitParams
from myogen.simulation.emg import EMGModel

r = REFS["hdemg"]
CACHE = Path.home() / ".cache" / "myogen" / "hyser"


def recorded() -> dict[str, np.ndarray]:
    """{grid: (N, 8, 8)} monopolar channels of the Hyser record during contraction (download once)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for ext in ("hea", "dat"):
        if not (CACHE / f"{r['record']}.{ext}").exists():
            urllib.request.urlretrieve(f"{r['url']}{r['record']}.{ext}", CACHE / f"{r['record']}.{ext}")
    names = [line.split()[-1] for line in (CACHE / f"{r['record']}.hea").read_text().splitlines()[1:]]
    x = np.fromfile(CACHE / f"{r['record']}.dat", "<i2").reshape(-1, r["channels"]).astype(float)
    env = signal.medfilt(np.sqrt((x ** 2).mean(1)), 2047)
    x = x[env > np.median(env)]                                         # the contractions, not the rest
    grids = {}
    for g in r["grids"]:
        rc = [tuple(int(v) - 1 for v in n.split("-")[1:]) for n in names if n.startswith(g)]
        idx = [i for i, n in enumerate(names) if n.startswith(g)]
        grids[g] = np.zeros((len(x), 8, 8))
        grids[g][:, [a for a, _ in rc], [b for _, b in rc]] = x[:, idx]
    return grids


def simulated(side: str, waveform: str, f: Forearm) -> np.ndarray:
    """(N, 8, 8) monopolar grid over the extensors or flexors: rows along the forearm, columns around it."""
    at = r["placement"][side]
    step_deg = np.degrees(r["ied_mm"] / f.subject.forearm_radius_mm)     # arc of one step (radius at the elbow)
    grid = mg.Electrodes([mg.Electrode(at["angle_deg"] + (j - 3.5) * step_deg, at["position_cm"] + (i - 3.5) * 1)
                          for i in range(8) for j in range(8)], derivation="monopolar")
    params = EMGParams(engine="motor_units", decay_mm=r["decay_mm"], motor_units=MotorUnitParams(waveform=waveform))
    x = EMGModel(f.model, f.subject, grid, params).generate(f.activity(r["drive"], 400), np.random.default_rng(0))
    band = signal.butter(4, r["band_hz"], btype="bandpass", fs=params.fs, output="sos")
    return signal.sosfiltfilt(band, x, axis=0).reshape(-1, 8, 8)


def correlation(G: np.ndarray, d: int, differential: bool) -> float:
    """Mean correlation of channels d grid steps apart along either grid axis; differential: of neighbour
    differences along that axis (the grid's orientation on the forearm is not in the record)."""
    out = []
    for axis in (1, 2):
        H = np.moveaxis(np.diff(G, axis=axis) if differential else G, axis, 1)
        out += [np.corrcoef(H[:, i, k], H[:, i + d, k])[0, 1] for i in range(H.shape[1] - d) for k in range(H.shape[2])]
    return float(np.mean(out))


f = Forearm()
real = recorded()
report = Report("Spatial spread on an 8x8 grid (10 mm): correlation vs distance, Hyser vs simulated")
for waveform in ("biphasic", "volume_conductor"):
    for side in ("extensor", "flexor"):
        sim = simulated(side, waveform, f)
        grids = [g for g, s in r["grids"].items() if s == side]
        for differential in (False, True):
            for d in r["distances"]:
                want = np.mean([correlation(real[g], d, differential) for g in grids])
                got = correlation(sim, d, differential)
                name = "differential" if differential else "monopolar"
                report.check(f"{waveform[:9]} {side} {name} {d * r['ied_mm']} mm (real {want:.2f})", got,
                             want - r["tolerance"], want + r["tolerance"])
report.show()
