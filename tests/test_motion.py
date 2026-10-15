"""The input: HandMotion and JointAdapter."""
import numpy as np
import pytest

import myogen as mg
from myogen.inputs.motion import DIP_PER_PIP, REST_POSTURE
from myogen.simulation.musculoskeletal import HAND_JOINTS


def test_model_names_pass_through_and_missing_joints_are_filled():
    t = np.linspace(0, 1, 100)
    motion = mg.JointAdapter()({"pm2_flexion_r": t, "flexion_r": -t}, fs=100)
    q = dict(zip(HAND_JOINTS, motion.angles.T))
    assert np.allclose(q["pm2_flexion_r"], t) and np.allclose(q["flexion_r"], -t)
    assert np.allclose(q["mcp2_flexion_r"], t)                       # MCP from PIP
    assert np.allclose(q["md2_flexion_r"], DIP_PER_PIP * t)          # DIP from PIP
    assert np.allclose(q["pm3_flexion_r"], REST_POSTURE["pm3_flexion_r"])
    assert "md2_flexion_r" in motion.filled and "pm2_flexion_r" not in motion.filled


def test_mapping_sign_offset_and_degrees():
    adapter = mg.JointAdapter({"wrist": ("flexion_r", -1.0, 0.1), "index_pip": "pm2_flexion_r"}, degrees=True)
    motion = adapter({"wrist": np.full(60, 30.0), "index_pip": np.full(60, 90.0)}, fs=60)
    q = dict(zip(HAND_JOINTS, motion.angles[0]))
    assert q["flexion_r"] == pytest.approx(-np.radians(30) + 0.1) and q["pm2_flexion_r"] == pytest.approx(np.pi / 2)


def test_unknown_joints_are_errors():
    with pytest.raises(ValueError, match="mapping"):
        mg.JointAdapter()({"index_pip": np.zeros(100)}, fs=100)
    with pytest.raises(ValueError, match="unknown model joints"):
        mg.JointAdapter({"x": "not_a_joint"})


def test_validation():
    good = mg.example_motion(1.0).angles
    with pytest.raises(ValueError, match="shape"):
        mg.HandMotion(good[:, :5], fs=100)
    with pytest.raises(ValueError, match="fs"):
        mg.HandMotion(good, fs=0)
    with pytest.raises(ValueError, match="NaN"):
        mg.HandMotion(np.where(np.arange(23) == 3, np.nan, good), fs=100)
    with pytest.raises(ValueError, match="long"):
        mg.HandMotion(good[:10], fs=100)


@pytest.mark.parametrize("suffix", [".h5", ".npz"])
def test_file_roundtrip(tmp_path, suffix):
    m = mg.example_motion(2.0, fs=50)
    m.elbow = np.linspace(1.0, 1.2, len(m.angles))
    back = mg.HandMotion.load(m.save(tmp_path / f"m{suffix}"), mg.read_motion_file)
    assert back.fs == 50 and np.allclose(back.angles, m.angles) and np.allclose(back.elbow, m.elbow)
    assert np.array_equal(back.rest, m.rest) and back.filled == ()


def test_custom_loader(tmp_path):
    m = mg.example_motion(2.0, fs=50)
    path = tmp_path / "trial.csv"
    np.savetxt(path, m.angles, delimiter=",")
    read = lambda p: {"angles": np.loadtxt(p, delimiter=","), "fs": 50.0}
    assert np.allclose(mg.HandMotion.load(path, read).angles, m.angles)
    assert mg.HandMotion.load(path, lambda p: m) is m                    # a loader may return a HandMotion
    with pytest.raises(ValueError, match="needs 'angles' and 'fs'"):
        mg.HandMotion.load(path, lambda p: {"angles": m.angles})
    with pytest.raises(ValueError, match="has no"):
        mg.HandMotion.load(path, lambda p: {"angles": m.angles, "fs": 50.0, "glove": 1})
    with pytest.raises(TypeError):
        mg.HandMotion.load(path, lambda p: m.angles)
    with pytest.raises(FileNotFoundError):
        mg.HandMotion.load(tmp_path / "missing.csv", read)


def test_resampling_keeps_duration():
    m = mg.example_motion(2.0, fs=60).resampled(100)
    assert m.angles.shape == (200, 23) and m.rest.shape == (200,)
