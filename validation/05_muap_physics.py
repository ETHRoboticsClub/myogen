import numpy as np
from common import REFS, Forearm, Report

import myogen as mg
from myogen.params import EMGParams, MotorUnitParams
from myogen.simulation.emg import EMGModel
from myogen.simulation.motor_units import REFERENCE_DISTANCE_M, _biphasic
from myogen.simulation.volume_conductor import MUAPTable

r = REFS["muap"]
f = Forearm()
emg = EMGModel(f.model, f.subject, mg.Electrodes([mg.Electrode(60, 7)]), EMGParams())
mp, fs, fat = MotorUnitParams(), emg.fs, emg.fat
skin = float(emg.skin_radius(0.07))
table = MUAPTable(mp.volume_conductor, skin, fat, fs)
half, delay = r["ied_mm"] / 2000, r["ied_mm"] / 1000 / EMGParams().conduction_velocity


def physics(depth: float, angle: float, bipolar: bool) -> np.ndarray:
    """Volume conductor MUAP at a pole REFERENCE_DISTANCE_M from the end-plate (or the pair around it)."""
    wave = lambda dz: table.weights_at(np.array(depth), np.array(angle), np.array(dz)) @ table.basis   # noqa: E731
    return wave(REFERENCE_DISTANCE_M + half) - wave(REFERENCE_DISTANCE_M - half) if bipolar else \
        wave(REFERENCE_DISTANCE_M)


def biphasic(depth: float, angle: float, bipolar: bool, decay: float) -> np.ndarray:
    """The biphasic engine's MUAP: a point unit under the electrode centre, gain exp(-distance / decay)."""
    unit = (skin - fat - depth) * np.array([np.cos(angle), np.sin(angle)])
    def pole(z: float, shift: float = 0.0) -> np.ndarray:
        dist = np.hypot(np.linalg.norm(unit - [skin, 0]), z)
        w = _biphasic(mp.width_ms / 1000 + dist * mp.width_per_mm, fs) * np.exp(-dist / decay)
        return np.interp(np.arange(-200, 201) - shift * fs, np.arange(len(w)) - len(w) // 2, w, left=0, right=0)
    return pole(-half) - pole(half, delay) if bipolar else pole(0.0)


def decay_mm(amps: np.ndarray, at_mm: np.ndarray) -> float:
    """Distance over which the peak-to-peak amplitude falls by 1/e (log-linear fit)."""
    return -1 / np.polyfit(at_mm, np.log(amps / amps[0]), 1)[0]


report = Report("MUAP to electrode: biphasic engine vs layered volume conductor (decay length ratio)")
depths, angles = np.array(r["depths_mm"]), np.radians(r["angles_deg"])
arc_mm = angles * skin * 1000
for decay in r["decays_mm"]:
    for bipolar in (False, True):
        name = f"decay {decay} mm, {'bipolar' if bipolar else 'monopolar'}"
        phys_d = [np.ptp(physics(d / 1000, 0.0, bipolar)) for d in depths]
        bip_d = [np.ptp(biphasic(d / 1000, 0.0, bipolar, decay / 1000)) for d in depths]
        phys_a = [np.ptp(physics(0.004, a, bipolar)) for a in angles]
        bip_a = [np.ptp(biphasic(0.004, a, bipolar, decay / 1000)) for a in angles]
        for axis, phys, bip, at in (("depth", phys_d, bip_d, depths), ("around", phys_a, bip_a, arc_mm)):
            p, b = decay_mm(np.array(phys), at), decay_mm(np.array(bip), at)
            report.check(f"{name}, {axis}: {b:.1f} / {p:.1f} mm", b / p, *r["ratio"])
report.show()
