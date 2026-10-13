"""Electrode positions: presets, files, pole geometry."""
import pytest

import myogen as mg


def test_presets():
    myo = mg.Electrodes.myo()
    assert len(myo) == 16 and myo.fs_out == 200
    assert myo.electrodes[8].angle_deg - myo.electrodes[0].angle_deg == pytest.approx(22.5)    # second band rotated
    assert len(mg.Electrodes.otto_bock()) == 10 and mg.Electrodes.otto_bock().fs_out == 100
    assert abs(mg.Electrodes.ring(n=4, rotation_deg=90).electrodes[1].angle_deg) == pytest.approx(180.0)


def test_yaml_roundtrip(tmp_path):
    e = mg.Electrodes([mg.Electrode(90, 6.0, ied_mm=15, orientation_deg=30, name="ext")], device="raw", fs=1000)
    assert mg.Electrodes.load(e.save(tmp_path / "e.yml")) == e


def test_poles_follow_orientation():
    radius = lambda z: 0.04
    (a0, z0), (a1, z1) = mg.Electrode(0, 10, ied_mm=20, orientation_deg=0).poles(radius)
    assert a0 == a1 == 0 and z1 - z0 == pytest.approx(0.02)                 # along the forearm
    (a0, z0), (a1, z1) = mg.Electrode(0, 10, ied_mm=20, orientation_deg=90).poles(radius)
    assert z0 == pytest.approx(z1) and (a1 - a0) * 0.04 == pytest.approx(0.02)   # around it


def test_bad_device():
    with pytest.raises(ValueError, match="device"):
        mg.Electrodes([mg.Electrode(0, 5)], device="emg9000")
