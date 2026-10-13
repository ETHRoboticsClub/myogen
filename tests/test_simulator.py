"""The pipeline with the analytic engine: shapes, determinism, files, electrode physics."""
import numpy as np
import pytest

import myogen as mg
from myogen.body import ForearmModel
from myogen.engines.analytic import AnalyticEngine
from myogen.muscles import MuscleActivity


@pytest.fixture(scope="module")
def recording():
    return mg.Simulator(mg.Electrodes.myo()).run(mg.example_motion(2.0, fs=60))


def test_shapes_and_quality(recording):
    assert recording.emg.shape == (400, 16) and recording.fs == 200
    assert recording.labels["q"].shape == (200, 24) and recording.qc["pass"]


def test_deterministic(recording):
    again = mg.Simulator(mg.Electrodes.myo()).run(mg.example_motion(2.0, fs=60))
    assert np.array_equal(again.emg, recording.emg)


def test_file_roundtrip(recording, tmp_path):
    back = mg.Recording.load(recording.save(tmp_path / "r.h5"))
    assert np.array_equal(back.emg, recording.emg) and back.electrodes == recording.electrodes
    assert back.subject == recording.subject and np.allclose(back.labels["q"], recording.labels["q"])


def test_electrode_off_the_forearm():
    with pytest.raises(ValueError, match="off the forearm"):
        mg.Simulator(mg.Electrodes([mg.Electrode(0, 40.0)]))
    amputee = mg.Subject(amputee=True, residual_length=0.5)
    with pytest.raises(ValueError, match="off the forearm"):
        mg.Simulator(mg.Electrodes([mg.Electrode(0, 20.0)]), subject=amputee)


def _rms(electrodes, muscle="EDC3_r"):
    model = ForearmModel()
    engine = AnalyticEngine()
    engine.prepare(model, mg.Subject(), electrodes)
    T, nu = 200, model.nu
    drive = np.zeros((T, nu))
    drive[:, model.muscle_names.index(muscle)] = 0.5
    q = np.tile(model.default_qpos, (T, 1))
    activity = MuscleActivity(100.0, drive, drive, np.zeros_like(drive), drive, np.ones((T, nu)), q, q, np.zeros((T, model.m.nv)),
                              np.zeros((T, model.m.nv)), np.zeros(T))
    return np.sqrt((engine.generate(activity, np.random.default_rng(0)).emg ** 2).mean(0))


def test_bipolar_physics():
    """Over the extensors (+90 deg): an electrode along the fibres sees more than one across them, a wider pole
    spacing more than a narrower one, and the opposite side of the forearm much less."""
    along, across, wide, palmar = _rms(mg.Electrodes([
        mg.Electrode(90, 7, ied_mm=20, orientation_deg=0), mg.Electrode(90, 7, ied_mm=20, orientation_deg=90),
        mg.Electrode(90, 7, ied_mm=40, orientation_deg=0), mg.Electrode(-90, 7, ied_mm=20)]))
    assert along > across and wide > along and palmar < 0.2 * along


def test_effort_reaches_the_drive_not_the_mechanics():
    """Effort is torque-neutral co-activation: it raises the EMG drive and leaves the motion unchanged."""
    from myogen.muscles import MuscleController
    model = ForearmModel()
    T = 150
    q = np.tile(model.default_qpos, (T, 1))
    model.prepare(mg.Subject(), q[0], q[0])
    effort = np.zeros((T, model.nu))
    effort[:, model.muscle_names.index("EDC3_r")] = 0.2
    args = (q, np.zeros((T, model.m.nv)), np.zeros(model.m.nv), 1.0)
    plain, effortful = MuscleController(model).track(*args), MuscleController(model).track(*args, effort=effort)
    assert np.array_equal(plain.q, effortful.q)
    i = model.muscle_names.index("EDC3_r")
    assert effortful.drive[:, i].min() >= 0.2 > plain.drive[:, i].max()
