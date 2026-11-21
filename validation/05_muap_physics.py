import numpy as np
from common import REFS, Forearm, Report

import myogen as mg
from myogen.params import EMGParams
from myogen.simulation.geometry import ForearmGeometry
from myogen.simulation.montage import Montage
from myogen.simulation.muap import REFERENCE_DISTANCE_M, BiphasicMUAP, LayeredMUAP

r = REFS["muap"]
f = Forearm()
geometry = ForearmGeometry(f.model, f.subject)
Z = 0.07                                                  # electrode centre, 7 cm from the elbow
electrodes = {False: mg.Electrodes([mg.Electrode(0, Z * 100)], derivation="monopolar"),
              True: mg.Electrodes([mg.Electrode(0, Z * 100, ied_mm=r["ied_mm"])])}
poles = {bipolar: geometry.poles(e).reshape(-1, 3) for bipolar, e in electrodes.items()}


def muap(model: BiphasicMUAP | LayeredMUAP, depth: float, angle: float, bipolar: bool) -> np.ndarray:
    """The channel's waveform of one unit at this depth below the muscle surface and angle from the electrode: under
    the electrode centre (biphasic: a point source), or with its end-plate REFERENCE_DISTANCE_M before it."""
    z = Z - (REFERENCE_DISTANCE_M if model.propagates else 0.0)
    radius = geometry.under_fat(np.array(z)) - depth
    unit = np.array([[radius * np.cos(angle), radius * np.sin(angle), z]])
    w = model.weights(unit, poles[bipolar])[0]                                   # (poles, kernels)
    n = max(len(k) for k in model.kernels)
    centred = np.array([np.pad(k, ((n - len(k)) // 2,)) for k in model.kernels])  # (kernels, n)
    return Montage(electrodes[bipolar], EMGParams(), 1.0, model.propagates)((w @ centred).T)[:, 0]


def decay_mm(model: BiphasicMUAP | LayeredMUAP, bipolar: bool, depths: bool) -> float:
    """Distance over which the peak-to-peak amplitude falls by 1/e, into depth or around the forearm."""
    if depths:
        at = np.array(r["depths_mm"], float)
        amps = [np.ptp(muap(model, d / 1000, 0.0, bipolar)) for d in at]
    else:
        angles = np.radians(r["angles_deg"])
        at = angles * float(geometry.skin_radius(Z)) * 1000
        amps = [np.ptp(muap(model, 0.004, a, bipolar)) for a in angles]
    return -1 / np.polyfit(at, np.log(np.array(amps) / amps[0]), 1)[0]


report = Report("MUAP to electrode: biphasic vs layered volume conductor (decay length ratio)")
layered = LayeredMUAP(geometry, EMGParams(), np.vstack(list(poles.values())))
for decay in r["decays_mm"]:
    biphasic = BiphasicMUAP(geometry, EMGParams(decay_mm=decay))
    for bipolar in (False, True):
        for depths in (True, False):
            b, p = decay_mm(biphasic, bipolar, depths), decay_mm(layered, bipolar, depths)
            name = f"decay {decay} mm, {'bipolar' if bipolar else 'monopolar'}, {'depth' if depths else 'around'}"
            report.check(f"{name}: {b:.1f} / {p:.1f} mm", b / p, *r["ratio"])
report.show()
