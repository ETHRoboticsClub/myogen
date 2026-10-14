"""The pipeline: shapes, determinism, files, electrode physics."""
import numpy as np
import pytest

import myogen as mg
from myogen.simulation.emg import EMGModel
from myogen.simulation.muscle_activity import MuscleActivity, MuscleController
from myogen.simulation.musculoskeletal import ForearmModel


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
    emg = EMGModel(model, mg.Subject(), electrodes)
    T, nu = 200, model.nu
    drive = np.zeros((T, nu))
    drive[:, model.muscle_names.index(muscle)] = 0.5
    q = np.tile(model.default_qpos, (T, 1))
    nv = model.m.nv
    activity = MuscleActivity(100.0, drive, drive, np.zeros_like(drive), drive, np.ones((T, nu)), q, q,
                              np.zeros((T, nv)), np.zeros((T, nv)), np.zeros(T))
    return np.sqrt((emg.generate(activity, np.random.default_rng(0)) ** 2).mean(0))


def test_bipolar_physics():
    """Over the extensors (+90 deg): an electrode along the fibres sees more than one across them, a wider pole
    spacing more than a narrower one, and the opposite side of the forearm much less."""
    along, across, wide, palmar = _rms(mg.Electrodes([
        mg.Electrode(90, 7, ied_mm=20, orientation_deg=0), mg.Electrode(90, 7, ied_mm=20, orientation_deg=90),
        mg.Electrode(90, 7, ied_mm=40, orientation_deg=0), mg.Electrode(-90, 7, ied_mm=20)]))
    assert along > across and wide > along and palmar < 0.2 * along


def test_effort_reaches_the_drive_not_the_mechanics():
    """Effort is torque-neutral co-activation: it raises the EMG drive and leaves the motion unchanged."""
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


def test_effort_is_directional():
    """Holding the wrist flexed co-activates mainly the wrist flexors, holding it extended mainly the extensors;
    the antagonists get Subject.antagonist_ratio of the agonists' effort."""
    from myogen.params import EffortParams
    from myogen.simulation.muscle_activity import effort
    model = ForearmModel()
    T, wrist = 300, model.m.jnt_qposadr[model.m.joint("flexion_r").id]
    q_rest = model.default_qpos.copy()
    flexors, extensors = ([model.muscle_names.index(n) for n in names]
                          for names in (("FCR_r", "FCU_r"), ("ECRB_r", "ECRL_r")))
    subject = mg.Subject(antagonist_ratio=0.25)
    for angle, agonists, antagonists in ((0.6, flexors, extensors), (-0.6, extensors, flexors)):
        q = np.tile(q_rest, (T, 1))
        q[:, wrist] = angle
        co, _ = effort(model, q, q_rest, subject, np.random.default_rng(0), EffortParams(), 100.0)
        held = co[T // 2]
        assert held[agonists].min() > 2 * held[antagonists].max() > 0


def test_effort_relaxes_at_rest_and_when_letting_go():
    """No co-activation where the motion is labelled rest; moving back towards the relaxed posture is letting go."""
    from myogen.params import EffortParams
    from myogen.simulation.muscle_activity import effort
    model = ForearmModel()
    T, wrist = 400, model.m.jnt_qposadr[model.m.joint("flexion_r").id]
    q_rest = model.default_qpos.copy()
    q = np.tile(q_rest, (T, 1))
    q[:, wrist] = 0.6 * np.sin(np.pi * np.arange(T) / T)          # out to 0.6 rad and back
    rest = np.zeros(T, bool)
    rest[-50:] = True
    p = EffortParams(cocontraction_episode_prob=0.0, lead_s=0.0)
    co, _ = effort(model, q, q_rest, mg.Subject(), np.random.default_rng(0), p, 100.0, rest=rest)
    assert co[-10:].max() < 1e-3 < co[T // 2].max()
    out, back = co[T // 4].max(), co[3 * T // 4].max()            # same posture, moving out vs moving back
    assert back < out


def test_near_field_rejects_distant_sources():
    """With a finite near field, a distant muscle reaches a channel less than with the purely propagating model."""
    from myogen.params import EMGParams
    model = ForearmModel()
    far = [mg.Simulator("myo", params=mg.SimParams().replace(emg={"near_field_mm": v})).emg for v in (np.inf, 5.0)]
    w = [e.pole_weights(model.default_qpos).reshape(model.nu, -1, 16, 3) for e in far]
    assert np.allclose(w[0][..., 2], 0)                             # infinite near field: nothing in phase
    assert EMGParams().near_field_mm == np.inf
    assert w[1][..., 2].sum() > 0 and np.allclose(w[1][..., 1] + w[1][..., 2], w[0][..., 1])
