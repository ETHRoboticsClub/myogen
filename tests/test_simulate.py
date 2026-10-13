"""The simulation interface: Motion (input), SimConfig (config), simulate (output)."""
import h5py
import numpy as np
import pytest

import myogen as mg
from myogen.simulation.config import TEMPLATE


def test_template_is_the_default(tmp_path):
    path = mg.SimConfig.write_default(tmp_path / "sim.yml")
    assert path.read_text() == TEMPLATE
    assert mg.SimConfig.load(path) == mg.SimConfig()
    with pytest.raises(FileExistsError):
        mg.SimConfig.write_default(path)


@pytest.mark.parametrize("data, message", [
    ({"engine": {"gian": 2}}, "engine.gian"),
    ({"engin": {}}, "engin"),
    ({"engine": {"name": "fem"}}, "engine.name"),
    ({"device": {"electrodes": {"levle": 0.2}}}, "levle"),
    ({"subject": {"residual_length": 0.5}}, "amputees only"),
    ({"subject": {"fat_mm": [8, 2]}}, "lo <= hi"),
    ({"session": {"load": 2}}, "session.load"),
    ({"seed": -1}, "seed"),
])
def test_config_errors_name_the_key(data, message):
    with pytest.raises(ValueError, match=message):
        mg.SimConfig.from_dict(data)


def test_config_shorthand_and_paths(tmp_path):
    assert mg.SimConfig(engine="biomime", device="myo").device.fs_out == 200
    (tmp_path / "sim.yml").write_text("engine:\n  biomime:\n    muap_bank: banks/b.h5\n")
    assert mg.SimConfig.load(tmp_path / "sim.yml").engine.biomime.muap_bank == str(tmp_path / "banks" / "b.h5")


def test_motion_validation():
    z = mg.example_motion(2.0).latent
    with pytest.raises(ValueError, match="shape"):
        mg.Motion(z[:, :9], fs=100)
    with pytest.raises(ValueError, match="fs"):
        mg.Motion(z, fs=0)
    bad = z.copy()
    bad[5, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        mg.Motion(bad, fs=100)
    bad = z.copy()
    bad[:, 7] = 1.5
    with pytest.raises(ValueError, match="wrist_flex"):
        mg.Motion(bad, fs=100)
    bad[:, 7] = 1.02                                         # within tolerance: clipped
    assert mg.Motion(bad, fs=100).latent[:, 7].max() == 1.0
    with pytest.raises(ValueError, match="long"):
        mg.Motion(z[:10], fs=100)


@pytest.mark.parametrize("suffix", [".h5", ".npz"])
def test_motion_roundtrip(tmp_path, suffix):
    m = mg.example_motion(2.0, fs=50)
    m.restimulus = (m.latent[:, 1] < 0.3).astype(np.int16)
    back = mg.Motion.load(m.save(tmp_path / f"m{suffix}"))
    assert back.fs == 50 and np.allclose(back.latent, m.latent) and np.array_equal(back.restimulus, m.restimulus)


def test_motion_column_order_is_checked(tmp_path):
    with h5py.File(tmp_path / "m.h5", "w") as f:
        f["latent"] = mg.example_motion(1.0).latent
        f.attrs["fs"] = 100
        f.attrs["latent_names"] = list(reversed(mg.LATENT_NAMES))
    with pytest.raises(ValueError, match="order"):
        mg.Motion.load(tmp_path / "m.h5")


def test_resampling_keeps_duration():
    z, restim = mg.Motion(mg.example_motion(2.0).latent, fs=100, restimulus=np.zeros(200)).resampled(60)
    assert z.shape == (120, 10) and restim.shape == (120,)


def test_simulate_end_to_end(tmp_path):
    cfg = mg.SimConfig(device="myo", seed=3)
    cfg.subject.fat_mm = 5
    cfg.session.elbow_angle_deg = [60, 70]
    motion = mg.example_motion(2.0, seed=1, fs=60)
    r = mg.simulate(motion, cfg, out=tmp_path / "emg.h5")
    assert r.emg.shape == (400, 16) and r.labels["q"].shape[0] == 200 and r.qc["finite"]
    assert r.subject_params["fat"] == pytest.approx(0.005)
    assert 60 <= np.degrees(r.session_params["elbow_angle"]) <= 70
    with h5py.File(tmp_path / "emg.h5") as f:
        assert f["emg"].attrs["fs"] == 200 and f["labels"].attrs["fs"] == 100
        assert np.array_equal(f["emg"][:], r.emg)
    again = mg.simulate(motion, cfg)                         # same motion + config + seed -> same recording
    assert np.array_equal(again.emg, r.emg)
