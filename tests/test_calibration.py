"""Calibration recovers device settings that generated a recording."""
import copy

import numpy as np
import pytest

import myogen as mg
from myogen.body import HAND_JOINTS
from myogen.motion import REST_POSTURE


def motion_with_rest(rest_s=2.0, move_s=4.0, fs=100.0):
    """A relaxed hand (labelled rest), then open-close grasps: noise is only identifiable where muscles rest."""
    relaxed = np.tile([REST_POSTURE[j] for j in HAND_JOINTS], (int(rest_s * fs), 1))
    moving = mg.example_motion(move_s, fs)
    return mg.HandMotion(np.vstack([relaxed, moving.angles]), fs,
                         rest=np.r_[np.ones(len(relaxed), bool), np.zeros(len(moving.angles), bool)])


@pytest.mark.parametrize("device, truth", [
    ("myo", {"gain": 2.5, "noise_uv": 8.0}),
    ("otto_bock", {"gain": 0.5, "noise_uv": 3.0, "smooth_hz": 2.5}),
])
def test_recovers_known_device_settings(device, truth):
    sim = mg.Simulator(device)
    motion = motion_with_rest()
    target = copy.deepcopy(sim.electrodes)
    target.gain, target.noise_uv = truth["gain"], truth["noise_uv"]
    if "smooth_hz" in truth:
        target.frontend.smooth_hz = truth["smooth_hz"]
    real = sim.record(sim.simulate_clean(motion), target).emg
    cal = mg.calibrate(sim, [motion], [real])
    for name, value in truth.items():
        assert cal.fitted[name][1] == pytest.approx(value, rel=0.2), (name, cal.fitted)
    assert abs(cal.report["amplitude"]["after"] / cal.report["amplitude"]["real"] - 1) < 0.05
    assert str(cal).startswith("statistic")


def test_recovers_armband_rotation():
    from myogen.calibration import rotated
    sim = mg.Simulator("myo")
    motion = motion_with_rest()
    truth = sim.with_electrodes(rotated(sim.electrodes, 90.0))
    real = truth.record(truth.simulate_clean(motion)).emg
    cal = mg.calibrate(sim, [motion], [real], fit_rotation=True, rotation_steps=8)
    assert cal.fitted["rotation_deg"][1] == pytest.approx(90.0)
    assert cal.report["channel_correlation"]["after"] > cal.report["channel_correlation"]["before"]
